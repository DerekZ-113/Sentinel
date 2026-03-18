import { useEffect, useState } from "react";
import { fetchHealth, fetchStats } from "./services/api";
import type { HealthResponse, StatsResponse } from "./services/api";
import OverviewCards from "./components/OverviewCards";
import AlertFeed from "./components/AlertFeed";
import TypeBreakdown from "./components/TypeBreakdown";
import SimulatePanel from "./components/SimulatePanel";
import ModelHealth from "./components/ModelHealth";
import DemoBanner from "./components/DemoBanner";
import FPRateChart from "./components/FPRateChart";

const NAV_ITEMS = [
  { id: "overview", label: "Overview" },
  { id: "alerts", label: "Recent Alerts" },
  { id: "breakdown", label: "Alerts by Type" },
  { id: "fp-trend", label: "FP Rate Trend" },
  { id: "simulate", label: "Simulate" },
  { id: "model-health", label: "Model Health" },
];

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState("overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    Promise.all([fetchHealth(), fetchStats()])
      .then(([h, s]) => {
        setHealth(h);
        setStats(s);
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        }
      },
      { threshold: 0.15, rootMargin: "-20% 0px -60% 0px" }
    );

    for (const item of NAV_ITEMS) {
      const el = document.getElementById(item.id);
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  }, [health, stats]);

  function scrollTo(id: string) {
    setActiveSection(id);
    setSidebarOpen(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center text-red-400 text-xl">
        API Error: {error}
      </div>
    );
  }

  if (!health || !stats) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center text-gray-400 text-xl">
        Loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex">
      {/* Mobile header */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-14 bg-gray-900/95 border-b border-gray-800 flex items-center px-4 z-20 backdrop-blur-sm">
        <button
          onClick={() => setSidebarOpen(true)}
          className="text-gray-400 hover:text-white text-xl"
          aria-label="Open menu"
        >
          &#9776;
        </button>
        <div className="ml-3 flex items-center gap-2">
          <div className="h-6 w-6 rounded bg-blue-600 flex items-center justify-center text-[10px] font-bold">
            S
          </div>
          <span className="font-bold text-sm">Sentinel</span>
        </div>
      </div>

      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/50 z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-screen w-52 bg-gray-900/80 border-r border-gray-800 flex-col z-40
          ${sidebarOpen ? "flex" : "hidden"} md:flex`}
      >
        {/* Logo */}
        <div className="px-4 py-5 border-b border-gray-800">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center text-sm font-bold">
              S
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight">Sentinel</h1>
              <p className="text-gray-500 text-[10px]">Fleet Alert Monitoring</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollTo(item.id)}
              className={`w-full px-3 py-2 rounded-lg text-[13px] transition-colors text-left ${
                activeSection === item.id
                  ? "bg-blue-600/15 text-blue-400"
                  : "text-gray-400 hover:bg-gray-800/60 hover:text-gray-200"
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {/* Status footer */}
        <div className="px-4 py-3 border-t border-gray-800">
          <div className="flex items-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                health.status === "healthy" ? "bg-emerald-400" : "bg-red-400"
              }`}
            />
            <span className="text-gray-500 text-[11px]">
              {health.status} · {health.model_features} features
            </span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="md:ml-52 ml-0 flex-1 px-4 md:px-8 py-6 pt-20 md:pt-6 space-y-6 max-w-[1200px]">
        <DemoBanner />

        <section id="overview">
          <OverviewCards stats={stats} />
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          <section id="alerts">
            <AlertFeed />
          </section>
          <section id="breakdown" className="lg:sticky lg:top-6">
            <TypeBreakdown byType={stats.by_type} />
          </section>
        </div>

        <section id="fp-trend">
          <FPRateChart />
        </section>

        <section id="simulate">
          <SimulatePanel />
        </section>

        <section id="model-health">
          <ModelHealth />
        </section>
      </main>
    </div>
  );
}

export default App;
