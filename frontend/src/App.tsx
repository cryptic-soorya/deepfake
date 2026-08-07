import { Routes, Route, Link } from "react-router-dom";
import Scanner from "./pages/Scanner";
import LiveSession from "./pages/LiveSession";
import IdentityPortal from "./pages/IdentityPortal";
import ReportViewer from "./pages/ReportViewer";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <nav className="flex gap-4 border-b border-gray-800 p-4">
        <Link to="/">Scanner</Link>
        <Link to="/live">Live Session</Link>
        <Link to="/identity">Identity Portal</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Scanner />} />
        <Route path="/live" element={<LiveSession />} />
        <Route path="/identity" element={<IdentityPortal />} />
        <Route path="/report/:scanId" element={<ReportViewer />} />
      </Routes>
    </div>
  );
}
