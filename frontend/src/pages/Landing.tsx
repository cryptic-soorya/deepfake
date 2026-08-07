import { useLayoutEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { gsap, ScrollTrigger } from "../lib/gsap";
import { useLenis } from "../lib/useLenis";
import GlitchText from "../components/hud/GlitchText";
import AnimatedCounter from "../components/hud/AnimatedCounter";
import CornerFrame from "../components/hud/CornerFrame";
import Waveform from "../components/hud/Waveform";

const STACK = [
  "SCRFD",
  "MEDIAPIPE FACE MESH",
  "ARCFACE",
  "MINIFASNET",
  "EFFICIENTNET-B4 / SBI",
  "UNIVFD",
  "AASIST + XLSR-53",
  "SYNCNET",
  "GRAD-CAM++",
];

const PIPELINE = [
  {
    step: "01",
    title: "DETECT",
    body: "Frame-level and temporal classifiers score visual, audio, and lip-sync channels independently — no single modality can carry a verdict alone.",
  },
  {
    step: "02",
    title: "VERIFY",
    body: "Face recognition and liveness anti-spoofing confirm the person behind the pixels is who they claim to be, live, not replayed.",
  },
  {
    step: "03",
    title: "EXPLAIN",
    body: "Every verdict ships with a Grad-CAM heatmap and a plain-language narrative — a score without a reason is not a verdict.",
  },
];

export default function Landing() {
  useLenis();
  const containerRef = useRef<HTMLDivElement>(null);
  const pipelineRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      gsap.utils.toArray<HTMLElement>(".pipeline-step").forEach((step) => {
        gsap.fromTo(
          step,
          { opacity: 0.15, y: 40 },
          {
            opacity: 1,
            y: 0,
            ease: "none",
            scrollTrigger: {
              trigger: step,
              start: "top 75%",
              end: "top 30%",
              scrub: true,
            },
          },
        );
      });

      gsap.utils.toArray<HTMLElement>(".reveal-up").forEach((el) => {
        gsap.fromTo(
          el,
          { opacity: 0, y: 24 },
          {
            opacity: 1,
            y: 0,
            duration: 0.8,
            ease: "power3.out",
            scrollTrigger: { trigger: el, start: "top 85%" },
          },
        );
      });

      ScrollTrigger.refresh();
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={containerRef} className="relative">
      {/* HERO */}
      <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 pt-20">
        <div className="mb-6 flex items-center gap-2 font-mono text-xs uppercase tracking-[0.3em] text-amber/80">
          <span className="h-1.5 w-1.5 animate-pulseDot rounded-full bg-amber" />
          Neurobots Championship 2026 // Prototype
        </div>
        <GlitchText
          as="h1"
          text="MORPHEUS.AI"
          className="text-glow-amber text-center font-display text-6xl font-semibold tracking-tight sm:text-8xl"
        />
        <p className="mt-6 max-w-xl text-center font-mono text-sm text-bone/60 sm:text-base">
          Real-time deepfake video, cloned-voice, and manipulated-identity detection —
          with identity verification and an explanation for every verdict.
        </p>
        <div className="mt-10 flex items-center gap-4">
          <Link
            to="/scan"
            className="border border-amber bg-amber/10 px-6 py-3 font-mono text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void"
          >
            Run a Scan &rarr;
          </Link>
          <Link
            to="/identity"
            className="border border-line-bright px-6 py-3 font-mono text-xs uppercase tracking-widest text-bone/70 transition-colors hover:border-bone/40 hover:text-bone"
          >
            Verify Identity
          </Link>
        </div>

        <div className="absolute bottom-10 flex flex-col items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-bone/30">
          <span>Scroll</span>
          <span className="h-8 w-px animate-pulse bg-bone/20" />
        </div>
      </section>

      {/* PIPELINE */}
      <section ref={pipelineRef} className="relative mx-auto max-w-4xl px-6 py-32">
        <p className="reveal-up mb-16 font-mono text-xs uppercase tracking-[0.3em] text-bone/40">
          Pipeline / How a verdict is formed
        </p>
        <div className="flex flex-col gap-24">
          {PIPELINE.map((p) => (
            <div key={p.step} className="pipeline-step grid grid-cols-[auto_1fr] gap-6 sm:gap-10">
              <span className="font-mono text-4xl text-amber/70 sm:text-6xl">{p.step}</span>
              <div className="border-l border-line pl-6 sm:pl-10">
                <h3 className="font-display text-3xl font-semibold sm:text-4xl">{p.title}</h3>
                <p className="mt-3 max-w-lg font-mono text-sm leading-relaxed text-bone/60">{p.body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* STATS */}
      <section className="reveal-up mx-auto grid max-w-4xl grid-cols-2 gap-8 px-6 py-24 sm:grid-cols-4">
        {[
          { label: "Model Stack", value: 10, suffix: "" },
          { label: "Modalities Fused", value: 3, suffix: "" },
          { label: "Target Latency", value: 300, suffix: "ms" },
          { label: "API Version", value: 1, suffix: "" },
        ].map((s) => (
          <CornerFrame key={s.label} className="p-6">
            <AnimatedCounter value={s.value} suffix={s.suffix} className="font-mono text-3xl text-amber" />
            <p className="mt-2 font-mono text-[10px] uppercase tracking-widest text-bone/40">{s.label}</p>
          </CornerFrame>
        ))}
      </section>

      {/* LIVE SIGNAL */}
      <section className="reveal-up mx-auto max-w-4xl px-6 py-24">
        <CornerFrame active className="flex flex-col items-center gap-6 p-10">
          <p className="font-mono text-xs uppercase tracking-widest text-bone/40">Audio channel / live signal preview</p>
          <Waveform bars={48} className="w-full justify-center" />
        </CornerFrame>
      </section>

      {/* MODEL STACK MARQUEE */}
      <section className="reveal-up overflow-hidden border-y border-line py-6">
        <div className="flex w-max animate-marquee gap-10 font-mono text-sm uppercase tracking-widest text-bone/30">
          {[...STACK, ...STACK].map((name, i) => (
            <span key={`${name}-${i}`} className="flex items-center gap-10">
              {name}
              <span className="text-amber/40">/</span>
            </span>
          ))}
        </div>
      </section>

      {/* CLOSING CTA */}
      <section className="reveal-up mx-auto flex max-w-3xl flex-col items-center px-6 py-32 text-center">
        <h2 className="font-display text-4xl font-semibold sm:text-5xl">Every verdict, explained.</h2>
        <p className="mt-4 max-w-lg font-mono text-sm text-bone/60">
          No black-box scores. Upload a video or image and see exactly what the model saw.
        </p>
        <Link
          to="/scan"
          className="mt-8 border border-amber bg-amber/10 px-8 py-3 font-mono text-xs uppercase tracking-widest text-amber transition-colors hover:bg-amber hover:text-void"
        >
          Start Scanning &rarr;
        </Link>
      </section>
    </div>
  );
}
