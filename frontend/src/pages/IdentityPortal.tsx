import { motion } from "framer-motion";
import CornerFrame from "../components/hud/CornerFrame";

const STEPS = [
  { num: "01", label: "Capture", desc: "Face detection + landmark alignment (SCRFD)" },
  { num: "02", label: "Liveness", desc: "Anti-spoofing pass (Silent-Face / MiniFASNet)" },
  { num: "03", label: "Match", desc: "Identity embedding + comparison (ArcFace)" },
];

export default function IdentityPortal() {
  return (
    <div className="mx-auto max-w-4xl px-6 pb-24 pt-32">
      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className="font-mono text-xs uppercase tracking-[0.3em] text-amber/70"
      >
        Module 03 // Identity
      </motion.p>
      <motion.h1
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="mt-2 font-display text-4xl font-semibold sm:text-5xl"
      >
        Identity Portal
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mt-3 max-w-xl font-mono text-sm text-bone/50"
      >
        Enroll and verify a person's identity via face match and liveness — confirms the person behind the pixels is real and present.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-10 grid gap-6 sm:grid-cols-[1fr_1.2fr]"
      >
        <CornerFrame active className="relative flex min-h-[20rem] items-center justify-center p-6">
          <div className="relative flex h-48 w-48 items-center justify-center">
            <span className="absolute -top-1 -left-1 h-6 w-6 border-l-2 border-t-2 border-amber" />
            <span className="absolute -top-1 -right-1 h-6 w-6 border-r-2 border-t-2 border-amber" />
            <span className="absolute -bottom-1 -left-1 h-6 w-6 border-l-2 border-b-2 border-amber" />
            <span className="absolute -bottom-1 -right-1 h-6 w-6 border-r-2 border-b-2 border-amber" />
            <span className="font-mono text-[10px] uppercase tracking-widest text-bone/30">Align Face</span>
          </div>
        </CornerFrame>

        <div className="flex flex-col gap-4">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.num}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 + i * 0.08 }}
              className="flex items-start gap-4 border border-line bg-ink/40 p-4"
            >
              <span className="font-mono text-lg text-amber/60">{s.num}</span>
              <div>
                <p className="font-display text-sm font-semibold text-bone">{s.label}</p>
                <p className="mt-1 font-mono text-xs text-bone/40">{s.desc}</p>
              </div>
            </motion.div>
          ))}
          <button className="mt-2 border border-amber bg-amber/10 px-6 py-2.5 font-mono text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void">
            Begin Enrollment &rarr;
          </button>
        </div>
      </motion.div>
    </div>
  );
}
