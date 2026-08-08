import { useState } from "react";
import clsx from "clsx";

export interface ScorePoint {
  t: number;
  score: number | null;
}

interface ScoreSparklineProps {
  points: ScorePoint[];
  maxPoints: number;
  className?: string;
}

const WIDTH = 480;
const HEIGHT = 120;
const PAD_Y = 10;

function statusColor(score: number | null): string {
  if (score === null) return "#4a4a4f";
  if (score >= 0.7) return "#ff3b30";
  if (score >= 0.4) return "#ffb020";
  return "#3ddc84";
}

export default function ScoreSparkline({ points, maxPoints, className }: ScoreSparklineProps) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const known = points.filter((p) => p.score !== null) as { t: number; score: number }[];
  const xFor = (i: number) => (points.length <= 1 ? 0 : (i / (maxPoints - 1)) * WIDTH);
  const yFor = (score: number) => HEIGHT - PAD_Y - score * (HEIGHT - PAD_Y * 2);

  const path = known
    .map((p, i) => {
      const idx = points.indexOf(p);
      return `${i === 0 ? "M" : "L"} ${xFor(idx).toFixed(1)} ${yFor(p.score).toFixed(1)}`;
    })
    .join(" ");

  const hovered = hoverIndex !== null ? points[hoverIndex] : null;

  return (
    <div className={clsx("relative", className)}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-28 w-full touch-none"
        preserveAspectRatio="none"
        onMouseLeave={() => setHoverIndex(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const relX = ((e.clientX - rect.left) / rect.width) * WIDTH;
          const idx = Math.round((relX / WIDTH) * (maxPoints - 1));
          setHoverIndex(Math.min(Math.max(idx, 0), points.length - 1));
        }}
      >
        {/* recessive midline at the 0.5 fake/real threshold */}
        <line x1={0} y1={yFor(0.5)} x2={WIDTH} y2={yFor(0.5)} stroke="#2a2a2e" strokeWidth={1} strokeDasharray="2 3" />

        {path && <path d={path} fill="none" stroke="#ffb020" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />}

        {known.map((p) => {
          const idx = points.indexOf(p);
          const isLast = idx === points.length - 1;
          if (!isLast) return null;
          return (
            <circle key={idx} cx={xFor(idx)} cy={yFor(p.score)} r={3.5} fill={statusColor(p.score)} />
          );
        })}

        {hovered && hovered.score !== null && (
          <line
            x1={xFor(hoverIndex!)}
            y1={PAD_Y}
            x2={xFor(hoverIndex!)}
            y2={HEIGHT - PAD_Y}
            stroke="#8a8a8f"
            strokeWidth={1}
          />
        )}
      </svg>

      {hovered && hovered.score !== null && (
        <div
          className="pointer-events-none absolute top-0 -translate-x-1/2 rounded-sm border border-line-bright bg-ink px-2 py-1 font-mono text-[10px] text-bone/80"
          style={{ left: `${(xFor(hoverIndex!) / WIDTH) * 100}%` }}
        >
          P(fake) {hovered.score.toFixed(2)}
        </div>
      )}
    </div>
  );
}
