import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import CornerFrame from "../components/hud/CornerFrame";
import StatusPill from "../components/StatusPill";
import { getScan, getExplanation, heatmapUrl, isTerminalStatus, type ScanResponse } from "../lib/api";

const MODEL_LABELS: Record<string, string> = {
  frame_classifier: "Visual",
  audio_deepfake: "Audio",
  lipsync: "Lip-Sync",
};

const PENDING_LABELS: Record<string, string> = {
  pending: "Queued",
  processing: "Analyzing",
};

export default function ReportViewer() {
  const { scanId } = useParams();
  const [scan, setScan] = useState<ScanResponse | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explanationStatus, setExplanationStatus] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const explanationRequested = useRef(false);
  const startedAt = useRef(Date.now());
  const pending = !scan || !isTerminalStatus(scan.status);

  useEffect(() => {
    startedAt.current = Date.now();
    setElapsedMs(0);
  }, [scanId]);

  useEffect(() => {
    if (!pending) return;
    const tick = setInterval(() => setElapsedMs(Date.now() - startedAt.current), 250);
    return () => clearInterval(tick);
  }, [pending]);

  useEffect(() => {
    if (!scanId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const result = await getScan(scanId);
        if (cancelled) return;
        setScan(result);
        setFetchError(null);

        if (!isTerminalStatus(result.status)) {
          timer = setTimeout(poll, 2000);
          return;
        }

        if (result.status === "completed" && !explanationRequested.current) {
          explanationRequested.current = true;
          const exp = await getExplanation(scanId);
          if (!cancelled) {
            setExplanation(exp.explanation);
            setExplanationStatus(exp.status);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setFetchError(err instanceof Error ? err.message : "failed to load scan");
          timer = setTimeout(poll, 3000);
        }
      }
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [scanId]);

  const status = scan?.status ?? "pending";
  const isPending = !isTerminalStatus(status);
  const frameResult = scan?.model_results.find((r) => r.model_name === "gradcam");
  const hasHeatmap = Boolean(frameResult?.metadata?.heatmap_key);
  const elapsedLabel = `${(elapsedMs / 1000).toFixed(1)}s`;

  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-mono text-xs uppercase tracking-[0.3em] text-amber/70"
      >
        Forensic Report
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mt-2 font-display text-4xl font-semibold sm:text-5xl"
      >
        Scan {scanId}
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-3 flex items-center gap-2 font-mono text-xs text-bone/30"
      >
        {isPending && <span className="h-1.5 w-1.5 rounded-full bg-amber animate-pulseDot" />}
        {fetchError
          ? `Retrying — ${fetchError}`
          : isPending
            ? `Status: ${PENDING_LABELS[status] ?? status} — running detection pipeline… ${elapsedLabel}`
            : `Status: ${status} — finished in ${elapsedLabel}`}
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-10 grid gap-6 sm:grid-cols-[1fr_1fr]"
      >
        <CornerFrame active={isPending} className="p-6">
          <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Fused Verdict</p>
          <div className="mt-3">
            <StatusPill
              score={scan?.fused_score ?? 0}
              label={scan?.fused_verdict ?? (isPending ? "Scanning" : status)}
              pending={isPending}
            />
          </div>
          <div className="mt-6 flex flex-col gap-3">
            {["frame_classifier", "audio_deepfake", "lipsync"].map((modelName) => {
              const r = scan?.model_results.find((m) => m.model_name === modelName);
              const done = r?.score != null;
              const notRun = r != null && r.score == null;
              return (
                <div key={modelName} className="flex items-center justify-between border-b border-line pb-2 font-mono text-xs">
                  <span className="flex items-center gap-2 text-bone/50">
                    {isPending && !r && (
                      <span className="h-1.5 w-1.5 rounded-full bg-amber/70 animate-pulseDot" />
                    )}
                    {MODEL_LABELS[modelName]}
                  </span>
                  <span className={done ? "text-bone/70" : notRun ? "text-bone/20" : "text-amber/70"}>
                    {done ? `${(r!.score! * 100).toFixed(1)}%` : notRun ? "—" : isPending ? "running…" : "—"}
                  </span>
                </div>
              );
            })}
          </div>
        </CornerFrame>

        <CornerFrame active={isPending} className="relative flex flex-col items-center justify-center gap-3 overflow-hidden p-6">
          <p className="self-start font-mono text-[10px] uppercase tracking-widest text-bone/40">Grad-CAM Heatmap</p>
          {hasHeatmap ? (
            <img
              src={heatmapUrl(scanId!)}
              alt="Grad-CAM heatmap"
              className="h-40 w-full border border-line-bright object-cover"
            />
          ) : (
            <div className="relative flex h-40 w-full items-center justify-center overflow-hidden border border-dashed border-line-bright font-mono text-[10px] uppercase tracking-widest text-bone/20">
              {isPending && (
                <div className="pointer-events-none absolute inset-x-0 top-0 h-16 animate-scan bg-gradient-to-b from-amber/20 to-transparent" />
              )}
              {isPending ? "Awaiting frame classification" : "No heatmap for this scan"}
            </div>
          )}
        </CornerFrame>

        <CornerFrame className="p-6 sm:col-span-2">
          <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Narrative Explanation</p>
          <p className="mt-3 font-mono text-sm leading-relaxed text-bone/70">
            {explanation
              ? explanation
              : explanationStatus === "blocked_on_model_integration"
                ? "Narrative explainer unavailable for this scan (Gemini not configured or unreachable)."
                : status === "completed"
                  ? "Generating narrative…"
                  : "Generated by Gemini once the fusion stage completes — walks through which signals drove the verdict in plain language."}
          </p>
        </CornerFrame>
      </motion.div>
    </div>
  );
}
