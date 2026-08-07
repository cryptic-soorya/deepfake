import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";
import Landing from "./pages/Landing";
import Scanner from "./pages/Scanner";
import LiveSession from "./pages/LiveSession";
import IdentityPortal from "./pages/IdentityPortal";
import ReportViewer from "./pages/ReportViewer";
import Nav from "./components/layout/Nav";
import PageTransition from "./components/layout/PageTransition";
import GridBackground from "./components/hud/GridBackground";
import ScanlineOverlay from "./components/hud/ScanlineOverlay";
import NoiseOverlay from "./components/hud/NoiseOverlay";

export default function App() {
  const location = useLocation();

  return (
    <div className="relative min-h-screen bg-void text-bone">
      <GridBackground />
      <NoiseOverlay />
      <ScanlineOverlay />
      <Nav />
      <main className="relative z-10">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route
              path="/"
              element={
                <Landing />
              }
            />
            <Route
              path="/scan"
              element={
                <PageTransition>
                  <Scanner />
                </PageTransition>
              }
            />
            <Route
              path="/live"
              element={
                <PageTransition>
                  <LiveSession />
                </PageTransition>
              }
            />
            <Route
              path="/identity"
              element={
                <PageTransition>
                  <IdentityPortal />
                </PageTransition>
              }
            />
            <Route
              path="/report/:scanId"
              element={
                <PageTransition>
                  <ReportViewer />
                </PageTransition>
              }
            />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  );
}
