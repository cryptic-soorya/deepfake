import clsx from "clsx";

interface GlitchTextProps {
  text: string;
  className?: string;
  as?: "h1" | "h2" | "span";
}

export default function GlitchText({ text, className, as = "h1" }: GlitchTextProps) {
  const Tag = as;
  return (
    <Tag className={clsx("group relative inline-block", className)}>
      <span className="relative z-10">{text}</span>
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 text-alert opacity-0 group-hover:opacity-70 group-hover:animate-glitch"
        style={{ transform: "translate(2px, 0)" }}
      >
        {text}
      </span>
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 text-signal opacity-0 group-hover:opacity-70 group-hover:animate-glitch"
        style={{ transform: "translate(-2px, 0)", animationDelay: "0.08s" }}
      >
        {text}
      </span>
    </Tag>
  );
}
