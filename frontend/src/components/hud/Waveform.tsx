import { motion } from "framer-motion";
import clsx from "clsx";

interface WaveformProps {
  bars?: number;
  active?: boolean;
  className?: string;
}

export default function Waveform({ bars = 32, active = true, className }: WaveformProps) {
  return (
    <div className={clsx("flex h-12 items-center gap-[3px]", className)}>
      {Array.from({ length: bars }).map((_, i) => {
        const base = 0.15 + Math.abs(Math.sin(i * 0.7)) * 0.85;
        return (
          <motion.span
            key={i}
            className={clsx("w-[3px] rounded-full", active ? "bg-amber" : "bg-line-bright")}
            initial={{ scaleY: 0.2 }}
            animate={active ? { scaleY: [base * 0.3, base, base * 0.4, base * 0.8, base * 0.3] } : { scaleY: 0.15 }}
            transition={{
              duration: 1.2 + (i % 5) * 0.15,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.02,
            }}
            style={{ height: "100%", transformOrigin: "center" }}
          />
        );
      })}
    </div>
  );
}
