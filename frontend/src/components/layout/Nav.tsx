import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";

const LINKS = [
  { to: "/scan", label: "Scanner", num: "01" },
  { to: "/live", label: "Live Session", num: "02" },
  { to: "/identity", label: "Identity", num: "03" },
];

export default function Nav() {
  const location = useLocation();

  return (
    <header className="fixed inset-x-0 top-0 z-[60] border-b border-line bg-void/70 backdrop-blur-md">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center gap-2 font-mono text-sm tracking-widest text-bone">
          <span className="h-2 w-2 animate-pulseDot rounded-full bg-amber" />
          MORPHEUS<span className="text-amber">.AI</span>
        </Link>
        <ul className="flex items-center gap-1">
          {LINKS.map((link) => {
            const isActive = location.pathname.startsWith(link.to);
            return (
              <li key={link.to} className="relative">
                <Link
                  to={link.to}
                  className={`relative flex items-center gap-2 px-4 py-2 font-mono text-xs uppercase tracking-widest transition-colors ${
                    isActive ? "text-amber" : "text-bone/50 hover:text-bone"
                  }`}
                >
                  <span className="text-[10px] opacity-60">{link.num}</span>
                  {link.label}
                </Link>
                {isActive && (
                  <motion.span
                    layoutId="nav-underline"
                    className="absolute inset-x-3 -bottom-[1px] h-px bg-amber"
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                  />
                )}
              </li>
            );
          })}
        </ul>
      </nav>
    </header>
  );
}
