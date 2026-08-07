export default function ScanlineOverlay() {
  return (
    <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden opacity-[0.06]">
      <div
        className="absolute inset-x-0 h-24 animate-scan bg-gradient-to-b from-transparent via-bone/60 to-transparent"
        style={{ animationDuration: "5s" }}
      />
    </div>
  );
}
