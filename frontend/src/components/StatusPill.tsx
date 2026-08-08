import { motion } from "framer-motion";
import clsx from "clsx";

interface StatusPillProps {
  score: number;
  label: string;
  pending?: boolean;
}

export default function StatusPill({ score, label, pending }: StatusPillProps) {
  const isFake = score > 0.5;
  // `score` is P(fake). Display it as confidence in whichever verdict we're
  // actually showing, so "REAL / 92%" reads as 92% confidence it's real,
  // not a leftover 92% fake-probability sitting next to a "real" label.
  const confidence = isFake ? score : 1 - score;

  return (
    <motion.span
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className={clsx(
        "inline-flex items-center gap-2 border px-3 py-1 font-mono text-xs uppercase tracking-widest",
        pending
          ? "border-amber/40 bg-amber/10 text-amber"
          : isFake
            ? "border-alert/40 bg-alert/10 text-alert"
            : "border-signal/40 bg-signal/10 text-signal",
      )}
    >
      <span
        className={clsx(
          "h-1.5 w-1.5 rounded-full animate-pulseDot",
          pending ? "bg-amber" : isFake ? "bg-alert" : "bg-signal",
        )}
      />
      {label}
      {!pending && (
        <>
          <span className="text-bone/50">/</span>
          {(confidence * 100).toFixed(1)}% confidence
        </>
      )}
    </motion.span>
  );
}
