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

export default function ReportViewer() {
  const { scanId } = useParams();
  const [scan, setScan] = useState<ScanResponse | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explanationStatus, setExplanationStatus] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const explanationRequested = useRef(false);

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
        className="mt-3 font-mono text-xs text-bone/30"
      >
        {fetchError
          ? `Retrying — ${fetchError}`
          : isPending
            ? `Status: ${status} — waiting on the detection pipeline…`
            : `Status: ${status}`}
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-10 grid gap-6 sm:grid-cols-[1fr_1fr]"
      >
        <CornerFrame className="p-6">
          <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Fused Verdict</p>
          <div className="mt-3">
            <StatusPill
              score={scan?.fused_score ?? 0}
              label={scan?.fused_verdict ?? (isPending ? "Pending" : status)}
            />
          </div>
          <div className="mt-6 flex flex-col gap-3">
            {["frame_classifier", "audio_deepfake", "lipsync"].map((modelName) => {
              const r = scan?.model_results.find((m) => m.model_name === modelName);
              return (
                <div key={modelName} className="flex items-center justify-between border-b border-line pb-2 font-mono text-xs">
                  <span className="text-bone/50">{MODEL_LABELS[modelName]}</span>
                  <span className={r?.score != null ? "text-bone/70" : "text-bone/20"}>
                    {r?.score != null ? `${(r.score * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
              );
            })}
          </div>
        </CornerFrame>

        <CornerFrame className="flex flex-col items-center justify-center gap-3 p-6">
          <p className="self-start font-mono text-[10px] uppercase tracking-widest text-bone/40">Grad-CAM Heatmap</p>
          {hasHeatmap ? (
            <img
              src={heatmapUrl(scanId!)}
              alt="Grad-CAM heatmap"
              className="h-40 w-full border border-line-bright object-cover"
            />
          ) : (
            <div className="flex h-40 w-full items-center justify-center border border-dashed border-line-bright font-mono text-[10px] uppercase tracking-widest text-bone/20">
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
