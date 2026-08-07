import { motion } from "framer-motion";
import clsx from "clsx";

interface StatusPillProps {
  score: number;
  label: string;
}

export default function StatusPill({ score, label }: StatusPillProps) {
  const isFake = score > 0.5;

  return (
    <motion.span
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className={clsx(
        "inline-flex items-center gap-2 border px-3 py-1 font-mono text-xs uppercase tracking-widest",
        isFake ? "border-alert/40 bg-alert/10 text-alert" : "border-signal/40 bg-signal/10 text-signal",
      )}
    >
      <span className={clsx("h-1.5 w-1.5 rounded-full animate-pulseDot", isFake ? "bg-alert" : "bg-signal")} />
      {label}
      <span className="text-bone/50">/</span>
      {(score * 100).toFixed(1)}%
    </motion.span>
  );
}
