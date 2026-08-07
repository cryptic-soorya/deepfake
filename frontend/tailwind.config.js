/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        void: "#050506",
        ink: "#0b0b0d",
        bone: "#e8e6e1",
        amber: {
          DEFAULT: "#ffb020",
          dim: "#8a5f14",
        },
        alert: {
          DEFAULT: "#ff3b30",
          dim: "#7a1f19",
        },
        signal: {
          DEFAULT: "#3ddc84",
          dim: "#1f6e42",
        },
        line: {
          DEFAULT: "#1c1c1f",
          bright: "#2a2a2e",
        },
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      keyframes: {
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        flicker: {
          "0%, 100%": { opacity: "1" },
          "92%": { opacity: "1" },
          "93%": { opacity: "0.4" },
          "94%": { opacity: "1" },
          "96%": { opacity: "0.6" },
          "97%": { opacity: "1" },
        },
        glitch: {
          "0%, 100%": { clipPath: "inset(0 0 0 0)", transform: "translate(0, 0)" },
          "20%": { clipPath: "inset(20% 0 60% 0)", transform: "translate(-2px, 1px)" },
          "40%": { clipPath: "inset(60% 0 5% 0)", transform: "translate(2px, -1px)" },
          "60%": { clipPath: "inset(10% 0 80% 0)", transform: "translate(-1px, 0)" },
          "80%": { clipPath: "inset(40% 0 30% 0)", transform: "translate(1px, 1px)" },
        },
        marquee: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        pulseDot: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.4", transform: "scale(0.85)" },
        },
      },
      animation: {
        scan: "scan 3s linear infinite",
        flicker: "flicker 6s linear infinite",
        marquee: "marquee 20s linear infinite",
        pulseDot: "pulseDot 1.6s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
