import type { ReactNode } from "react";
import clsx from "clsx";

interface CornerFrameProps {
  children: ReactNode;
  className?: string;
  active?: boolean;
}

export default function CornerFrame({ children, className, active = false }: CornerFrameProps) {
  const bracketColor = active ? "border-amber" : "border-line-bright";

  return (
    <div className={clsx("relative", className)}>
      <span className={clsx("absolute -top-px -left-px h-4 w-4 border-l border-t", bracketColor)} />
      <span className={clsx("absolute -top-px -right-px h-4 w-4 border-r border-t", bracketColor)} />
      <span className={clsx("absolute -bottom-px -left-px h-4 w-4 border-l border-b", bracketColor)} />
      <span className={clsx("absolute -bottom-px -right-px h-4 w-4 border-r border-b", bracketColor)} />
      <div className="border border-line bg-ink/60">{children}</div>
    </div>
  );
}
