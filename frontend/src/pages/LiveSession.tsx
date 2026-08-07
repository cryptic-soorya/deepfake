import { motion } from "framer-motion";
import CornerFrame from "../components/hud/CornerFrame";
import Waveform from "../components/hud/Waveform";

export default function LiveSession() {
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
        Real-time webcam deepfake detection over WebSocket, targeting sub-300ms updates. Session streaming is a Round 3 build target.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-10 grid gap-6 sm:grid-cols-[1.4fr_1fr]"
      >
        <CornerFrame className="flex min-h-[22rem] flex-col items-center justify-center gap-6 p-6">
          <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-bone/40">
            <span className="h-1.5 w-1.5 rounded-full bg-line-bright" />
            Awaiting Session
          </div>
          <div className="flex h-40 w-40 items-center justify-center border border-dashed border-line-bright font-mono text-[10px] uppercase tracking-widest text-bone/30">
            No feed
          </div>
          <p className="font-mono text-xs text-bone/30">Connect a webcam session to begin streaming inference.</p>
        </CornerFrame>

        <div className="flex flex-col gap-6">
          <CornerFrame className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Connection</p>
            <p className="mt-2 font-mono text-sm text-alert">DISCONNECTED</p>
          </CornerFrame>
          <CornerFrame className="p-6">
            <p className="font-mono text-[10px] uppercase tracking-widest text-bone/40">Audio Channel</p>
            <Waveform active={false} bars={20} className="mt-3" />
          </CornerFrame>
        </div>
      </motion.div>
    </div>
  );
}
