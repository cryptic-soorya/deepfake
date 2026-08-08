import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import CornerFrame from "../components/hud/CornerFrame";
import Waveform from "../components/hud/Waveform";
import ScoreSparkline, { type ScorePoint } from "../components/hud/ScoreSparkline";

type ConnectionState = "idle" | "connecting" | "live" | "stopped" | "error";

const SAMPLE_INTERVAL_MS = 500; // ~2fps, matches the batch pipeline's frame sampler
const AUDIO_CHUNK_MS = 4000; // AASIST expects ~4s of audio per prediction (64600 samples @ 16kHz)
const MAX_POINTS = 40; // 20s of rolling history at 2fps
// The headline "Verdict" is a per-frame EfficientNet-B4 score with no
// temporal model behind it (CLAUDE.md's GRU/temporal-attention head isn't
// wired into the live path) -- a single noisy/blurry/badly-lit frame can
// swing well past the 0.7 threshold on its own. Smoothing over the last few
// frames (~this many * SAMPLE_INTERVAL_MS) instead of showing the latest raw
// frame is what keeps one bad frame from flipping the whole verdict to
// "LIKELY FAKE". Widened from 6->10 (3s->5s) -- 6 was still letting single
// noisy detections (independent per-frame face crop, no tracking) flip the
// headline label; 10 trades a little responsiveness for a lot less flicker.
const VERDICT_SMOOTHING_WINDOW = 10;
// Session-wide certainty is a bounded trailing window per modality, not an
// ever-growing average -- the first samples of a session (captured while the
// webcam is still auto-exposing/focusing, right as the model finishes
// loading) used to stay baked into the average for the rest of the session
// with equal weight forever, so a rough start could keep reading "fake"
// long after the feed had settled. Video and audio are windowed separately
// (see SESSION_WINDOW_AUDIO) because they arrive at very different rates.
const SESSION_WINDOW_VIDEO = 60; // 30s at ~2fps
// Audio chunks are ~4s each, so 8 of them span roughly the same 30s as the
// video window above. Previously both modalities were pooled into one
// SESSION_WINDOW-sized array; since video arrives 8x more often than audio,
// that pool was ~90% video by sample count, so "Overall Certainty" was
// effectively just the video verdict relabeled -- it barely moved on a
// cloned-voice-only chunk, which read as audio/video "disagreeing" even
// though the individual channel readouts were both correct. Weighting each
// modality's own mean (see combinedCertainty) instead of pooling raw samples
// fixes that.
const SESSION_WINDOW_AUDIO = 8;
// Mirrors app/fusion/scoring.py's _MODEL_WEIGHTS for frame_classifier/
// audio_deepfake (lipsync isn't wired into the live path). Renormalized over
// whichever modality actually has samples, same as that module does.
const MODALITY_WEIGHTS = { video: 0.45, audio: 0.3 };
const FRAME_WIDTH = 480;
const FRAME_HEIGHT = 360;

// Matches app/routers/stream.py's TAG_VIDEO / TAG_AUDIO — a 1-byte modality
// tag prefixed onto every binary WS message so one socket carries both streams.
const TAG_VIDEO = 0x01;
const TAG_AUDIO = 0x02;

interface StreamMessage {
  session_id: string;
  modality: "video" | "audio";
  status: "ok" | "no_speech" | "blocked_on_model_integration";
  score: number | null;
  confidence: number | null;
}

function verdictLabel(score: number | null): { label: string; className: string } {
  if (score === null) return { label: "NO SIGNAL", className: "text-bone/40" };
  if (score >= 0.7) return { label: "LIKELY FAKE", className: "text-alert" };
  if (score >= 0.4) return { label: "UNCERTAIN", className: "text-amber" };
  return { label: "LIKELY AUTHENTIC", className: "text-signal" };
}

interface ScoredSample {
  score: number;
  confidence: number | null;
}

// Confidence-weighted mean over a trailing per-modality window — "no_speech"
// / null-score chunks are never added to these arrays, so silent stretches
// don't drag the session verdict toward "fake". Bounded rather than a
// whole-session average so early rough readings (webcam still auto-exposing
// while the model finished loading) age out instead of anchoring the verdict
// for the rest of the session.
function overallCertainty(samples: ScoredSample[]): { score: number | null; count: number } {
  if (samples.length === 0) return { score: null, count: 0 };
  let weightedSum = 0;
  let weightTotal = 0;
  for (const s of samples) {
    const w = s.confidence ?? 1;
    weightedSum += s.score * w;
    weightTotal += w;
  }
  return { score: weightTotal > 0 ? weightedSum / weightTotal : null, count: samples.length };
}

// Combines the two modalities' own means using MODALITY_WEIGHTS, renormalized
// over whichever modality actually has samples right now — same pattern as
// app/fusion/scoring.py's fuse(). This is what keeps a fast stream of video
// frames from drowning out a much slower stream of audio chunks in the
// headline "Overall Certainty" number (see SESSION_WINDOW_AUDIO comment).
function combinedCertainty(
  video: { score: number | null; count: number },
  audio: { score: number | null; count: number },
): { score: number | null; count: number } {
  const parts: { score: number; weight: number }[] = [];
  if (video.score !== null) parts.push({ score: video.score, weight: MODALITY_WEIGHTS.video });
  if (audio.score !== null) parts.push({ score: audio.score, weight: MODALITY_WEIGHTS.audio });
  const count = video.count + audio.count;
  if (parts.length === 0) return { score: null, count };
  const totalWeight = parts.reduce((sum, p) => sum + p.weight, 0);
  const score = parts.reduce((sum, p) => sum + p.score * p.weight, 0) / totalWeight;
  return { score, count };
}

// Median of the last `window` scored video points, ignoring nulls. Median
// (not mean) so a single outlier frame can't drag the headline verdict past
// the threshold on its own the way a mean would.
function smoothedScore(points: ScorePoint[], window: number): number | null {
  const recent = points.slice(-window).filter((p): p is { t: number; score: number } => p.score !== null);
  if (recent.length === 0) return null;
  const sorted = [...recent].map((p) => p.score).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

export default function LiveSession() {
  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [points, setPoints] = useState<ScorePoint[]>([]);
  const [latest, setLatest] = useState<StreamMessage | null>(null);
  const [latestAudio, setLatestAudio] = useState<StreamMessage | null>(null);
  const [blockedNotice, setBlockedNotice] = useState(false);
  const [videoSessionSamples, setVideoSessionSamples] = useState<ScoredSample[]>([]);
  const [audioSessionSamples, setAudioSessionSamples] = useState<ScoredSample[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioLoopActiveRef = useRef(false);

  const stopSession = useCallback((nextState: ConnectionState = "stopped") => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    audioLoopActiveRef.current = false;
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setConnection(nextState);
  }, []);

  const recordAudioChunk = useCallback((audioStream: MediaStream): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      const chunks: Blob[] = [];
      let recorder: MediaRecorder;
      try {
        recorder = new MediaRecorder(audioStream, { mimeType: "audio/webm;codecs=opus" });
      } catch (err) {
        reject(err);
        return;
      }
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      recorder.onstop = () => resolve(new Blob(chunks, { type: "audio/webm" }));
      recorder.onerror = (e) => reject(e);
      recorder.start();
      setTimeout(() => {
        if (recorder.state !== "inactive") recorder.stop();
      }, AUDIO_CHUNK_MS);
    });
  }, []);

  const runAudioLoop = useCallback(
    async (audioStream: MediaStream, ws: WebSocket) => {
      audioLoopActiveRef.current = true;
      while (audioLoopActiveRef.current) {
        let blob: Blob;
        try {
          blob = await recordAudioChunk(audioStream);
        } catch {
          return; // MediaRecorder unsupported for this mime type/track — drop the audio channel silently
        }
        if (!audioLoopActiveRef.current || ws.readyState !== WebSocket.OPEN) return;
        ws.send(new Blob([new Uint8Array([TAG_AUDIO]), blob]));
      }
    },
    [recordAudioChunk],
  );

  const startSession = useCallback(async () => {
    setErrorMsg(null);
    setBlockedNotice(false);
    setPoints([]);
    setLatest(null);
    setLatestAudio(null);
    setVideoSessionSamples([]);
    setAudioSessionSamples([]);
    setConnection("connecting");

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: FRAME_WIDTH, height: FRAME_HEIGHT },
        audio: true,
      });
    } catch {
      setErrorMsg("camera access denied or unavailable");
      setConnection("error");
      return;
    }
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }

    const sessionId = crypto.randomUUID();
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${wsProtocol}://${window.location.host}/api/v1/stream/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnection("live");
      intervalRef.current = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || ws.readyState !== WebSocket.OPEN) return;
        canvas.width = FRAME_WIDTH;
        canvas.height = FRAME_HEIGHT;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        canvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              ws.send(new Blob([new Uint8Array([TAG_VIDEO]), blob]));
            }
          },
          "image/jpeg",
          0.7,
        );
      }, SAMPLE_INTERVAL_MS);

      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length > 0) {
        runAudioLoop(new MediaStream(audioTracks), ws);
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg: StreamMessage = JSON.parse(event.data);
        if (msg.status === "blocked_on_model_integration") setBlockedNotice(true);
        // "no_speech" chunks carry a null score by design (see backend
        // audio_deepfake.py's silence gate) — they're excluded from both the
        // per-modality display and the session-wide certainty average rather
        // than counted as a low-confidence "fake" reading.
        if (msg.status === "ok" && msg.score !== null) {
          const sample = { score: msg.score as number, confidence: msg.confidence };
          if (msg.modality === "audio") {
            setAudioSessionSamples((prev) => {
              const next = [...prev, sample];
              return next.length > SESSION_WINDOW_AUDIO ? next.slice(next.length - SESSION_WINDOW_AUDIO) : next;
            });
          } else {
            setVideoSessionSamples((prev) => {
              const next = [...prev, sample];
              return next.length > SESSION_WINDOW_VIDEO ? next.slice(next.length - SESSION_WINDOW_VIDEO) : next;
            });
          }
        }
        if (msg.modality === "audio") {
          setLatestAudio(msg);
          return;
        }
        setLatest(msg);
        setPoints((prev) => {
          const next = [...prev, { t: Date.now(), score: msg.score }];
          return next.length > MAX_POINTS ? next.slice(next.length - MAX_POINTS) : next;
        });
      } catch {
        // ignore malformed frame
      }
    };

    ws.onerror = () => {
      setErrorMsg("stream connection error");
      stopSession("error");
    };

    ws.onclose = () => {
      setConnection((prev) => (prev === "live" ? "stopped" : prev));
    };
  }, [stopSession, runAudioLoop]);

  useEffect(() => () => stopSession("stopped"), [stopSession]);

  const verdict = verdictLabel(smoothedScore(points, VERDICT_SMOOTHING_WINDOW));
  const isLive = connection === "live";
  const session = combinedCertainty(overallCertainty(videoSessionSamples), overallCertainty(audioSessionSamples));
  const sessionVerdict = verdictLabel(session.score);
  const audioIsSilent = latestAudio?.status === "no_speech";

  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-mono text-xs uppercase tracking-[0.3em] text-amber/70"
      >
        Module 02 // Live
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mt-2 font-display text-4xl font-semibold sm:text-5xl"
      >
        Live Session
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-3 max-w-xl font-mono text-sm text-bone/50"
      >
        Real-time webcam deepfake detection over WebSocket, sampled at ~2fps and scored frame-by-frame.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-10 grid gap-6 sm:grid-cols-[1.4fr_1fr]"
      >
        <CornerFrame active={isLive} className="flex min-h-[22rem] flex-col items-center justify-center gap-6 p-6">
          {connection === "idle" || connection === "error" || connection === "stopped" ? (
            <>
              <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-bone/40">
                <span className="h-1.5 w-1.5 rounded-full bg-line-bright" />
                {connection === "error" ? "Session Error" : "Awaiting Session"}
              </div>
              <div className="flex h-40 w-40 items-center justify-center border border-dashed border-line-bright font-mono text-[10px] uppercase tracking-widest text-bone/30">
                No feed
              </div>
              {errorMsg && <p className="font-mono text-xs text-alert">{errorMsg}</p>}
              <button
                onClick={startSession}
                className="border border-line-bright px-4 py-2 font-mono text-xs uppercase tracking-widest text-bone/70 transition hover:border-amber hover:text-amber"
              >
                Start Session
              </button>
            </>
          ) : (
            <div className="w-full">
              <video ref={videoRef} muted playsInline className="w-full -scale-x-100 rounded-sm border border-line-bright" />
              <div className="mt-4 flex items-center justify-between">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Verdict</p>
                  <p className={`font-mono text-lg ${verdict.className}`}>{verdict.label}</p>
                </div>
                <button
                  onClick={() => stopSession("stopped")}
                  className="border border-line-bright px-3 py-1.5 font-mono text-xs uppercase tracking-widest text-bone/70 transition hover:border-alert hover:text-alert"
                >
                  Stop
                </button>
              </div>
            </div>
          )}
          <canvas ref={canvasRef} className="hidden" />
        </CornerFrame>

        <div className="flex flex-col gap-6">
          <CornerFrame className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Overall Certainty — trailing ~30s, video + audio weighted</p>
            <div className="mt-2 flex items-baseline justify-between">
              <p className={`font-mono text-lg ${sessionVerdict.className}`}>{sessionVerdict.label}</p>
              <p className="font-mono text-xs text-bone/40">
                {session.score !== null ? session.score.toFixed(2) : "—"} · {session.count} samples
              </p>
            </div>
          </CornerFrame>
          <CornerFrame className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Connection</p>
            <p
              className={`mt-2 font-mono text-sm ${
                connection === "live" ? "text-signal" : connection === "connecting" ? "text-amber" : "text-alert"
              }`}
            >
              {connection === "live" && "LIVE"}
              {connection === "connecting" && "CONNECTING..."}
              {(connection === "idle" || connection === "stopped") && "DISCONNECTED"}
              {connection === "error" && "ERROR"}
            </p>
          </CornerFrame>
          <CornerFrame className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">P(fake) — rolling 20s</p>
            <ScoreSparkline points={points} maxPoints={MAX_POINTS} className="mt-3" />
            <p className="mt-1 text-right font-mono text-xs text-bone/40">
              {latest?.score !== null && latest?.score !== undefined ? latest.score.toFixed(2) : "—"}
            </p>
          </CornerFrame>
          <CornerFrame className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Audio Channel</p>
            <Waveform active={isLive} bars={20} className="mt-3" />
            <div className="mt-2 flex items-center justify-between">
              <p className="font-mono text-[10px] text-bone/30">
                {audioIsSilent ? "no speech detected" : "P(fake), ~4s window"}
              </p>
              {audioIsSilent ? (
                <p className="font-mono text-xs text-bone/40">SILENT</p>
              ) : (
                <p className={`font-mono text-xs ${verdictLabel(latestAudio?.score ?? null).className}`}>
                  {latestAudio?.score !== null && latestAudio?.score !== undefined
                    ? latestAudio.score.toFixed(2)
                    : "—"}
                </p>
              )}
            </div>
          </CornerFrame>
        </div>
      </motion.div>

      {blockedNotice && (
        <p className="mt-6 font-mono text-xs text-bone/40">
          Backend reports <span className="text-amber">blocked_on_model_integration</span> for one or more frames —
          the frame classifier didn't load or didn't find a face; see stream.py.
        </p>
      )}
    </div>
  );
}
