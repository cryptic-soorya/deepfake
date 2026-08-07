interface VerdictBadgeProps {
  score: number;
  label: string;
}

export default function VerdictBadge({ score, label }: VerdictBadgeProps) {
  const color = score > 0.5 ? "bg-red-600" : "bg-green-600";
  return (
    <span className={`rounded px-2 py-1 text-sm text-white ${color}`}>
      {label}: {(score * 100).toFixed(1)}%
    </span>
  );
}
