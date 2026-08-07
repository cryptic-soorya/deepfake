export default function GridBackground() {
  return (
    <div
      className="pointer-events-none fixed inset-0 z-0 bg-grid"
      style={{
        maskImage: "radial-gradient(ellipse 80% 60% at 50% 0%, black 40%, transparent 100%)",
        WebkitMaskImage: "radial-gradient(ellipse 80% 60% at 50% 0%, black 40%, transparent 100%)",
      }}
    />
  );
}
