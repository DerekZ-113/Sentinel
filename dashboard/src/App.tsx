import { useCallback, useEffect, useState } from "react";
import { DEMO_MODE } from "./services/api";
import { useLiveBootstrap } from "./hooks/useLiveBootstrap";
import { useDemoBootstrap } from "./demo/demoData";
import OverviewCards from "./components/OverviewCards";
import AlertFeed from "./components/AlertFeed";
import LiveAlertFeed from "./components/LiveAlertFeed";
import TypeBreakdown from "./components/TypeBreakdown";
import ModelHealth from "./components/ModelHealth";
import DemoModelHealth from "./components/demo/DemoModelHealth";
import FPRateChart from "./components/FPRateChart";
import DemoFPRateChart from "./components/demo/DemoFPRateChart";
import StatusBar from "./components/StatusBar";
import LiveTopBar from "./components/LiveTopBar";
import AlertDetailDrawer from "./components/AlertDetailDrawer";
import SimulateDrawer from "./components/SimulateDrawer";
import LiveSimulateDrawer from "./components/LiveSimulateDrawer";
import { IconX } from "./components/icons";
import type { DemoAlert } from "./demo/types";

// Module-scope selection: DEMO_MODE is a build-time constant, so Rollup
// dead-code-eliminates the unused branch (and the demo modules + fixtures
// behind it) from live builds.
const useBootstrap = DEMO_MODE ? useDemoBootstrap : useLiveBootstrap;

const REFRESH_INTERVAL_MS = 5000;

function formatLastUpdated(date: Date | null) {
  if (!date) return null;
  return date.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
}

function App() {
  const { health, stats, error, refreshError, lastUpdatedAt, retry, refresh } =
    useBootstrap();
  const [liveRefreshEnabled, setLiveRefreshEnabled] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);
  const [selectedAlert, setSelectedAlert] = useState<DemoAlert | null>(null);
  const [simulateOpen, setSimulateOpen] = useState(false);
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [healthDetailOpen, setHealthDetailOpen] = useState(false);

  const ready = health !== null && stats !== null;

  // Live-mode polling: bump the token so data-owning panels refetch too
  const refreshDashboard = useCallback(() => {
    setRefreshToken((token) => token + 1);
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (DEMO_MODE || !liveRefreshEnabled || !ready) return;

    const intervalId = window.setInterval(refreshDashboard, REFRESH_INTERVAL_MS);
    return () => window.clearInterval(intervalId);
  }, [liveRefreshEnabled, ready, refreshDashboard]);

  function toggleLiveRefresh() {
    const nextEnabled = !liveRefreshEnabled;
    setLiveRefreshEnabled(nextEnabled);
    if (nextEnabled && ready) {
      refreshDashboard();
    }
  }

  function toggleTypeFilter(notificationType: string) {
    setSelectedType((current) =>
      current === notificationType ? null : notificationType
    );
  }

  const lastUpdatedText = formatLastUpdated(lastUpdatedAt);

  if (error) {
    return (
      <div className="min-h-screen bg-canvas flex flex-col items-center justify-center gap-4">
        <p className="text-crit text-xl">API Error: {error}</p>
        <button
          onClick={retry}
          className="text-[11px] uppercase tracking-[0.1em] text-crit hover:bg-crit/10 border border-crit/50 px-4 py-2 rounded-xs transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!health || !stats) {
    return (
      <div className="min-h-screen bg-canvas flex items-center justify-center text-ink-mid text-xl">
        Loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen lg:h-screen lg:overflow-hidden bg-canvas text-ink flex flex-col">
      {DEMO_MODE ? (
        <StatusBar health={health} onSimulate={() => setSimulateOpen(true)} />
      ) : (
        <LiveTopBar
          health={health}
          liveRefreshEnabled={liveRefreshEnabled}
          onToggleLiveRefresh={toggleLiveRefresh}
          lastUpdatedText={lastUpdatedText}
          refreshError={refreshError}
          onSimulate={() => setSimulateOpen(true)}
        />
      )}

      <main className="flex-1 min-h-0 flex flex-col gap-3 px-4 md:px-6 py-3 w-full max-w-[1600px] mx-auto">
        <OverviewCards
          stats={stats}
          windowLabel={DEMO_MODE ? "Current shift" : undefined}
        />

        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[62fr_38fr] gap-4">
          <section className="min-h-0 flex flex-col">
            {selectedType && (
              <div className="mb-2 flex items-center gap-2">
                <span className="text-ink-micro text-[10px] uppercase tracking-[0.1em]">
                  filtered:
                </span>
                <button
                  onClick={() => setSelectedType(null)}
                  aria-label="Clear type filter"
                  className="flex items-center gap-1.5 text-accent border border-accent/40 bg-accent/14 rounded-xs px-2.5 py-0.5 text-[10px] uppercase tracking-[0.05em] hover:bg-accent/20 transition-colors"
                >
                  {selectedType.replace(/_/g, " ")}
                  <IconX size={8} />
                </button>
              </div>
            )}
            {DEMO_MODE ? (
              <LiveAlertFeed
                key={selectedType ?? "all"}
                onSelect={setSelectedAlert}
                filterType={selectedType}
              />
            ) : (
              <AlertFeed
                key={selectedType ?? "all"}
                refreshToken={refreshToken}
                filterType={selectedType}
              />
            )}
          </section>
          <div className="min-h-0 lg:overflow-y-auto space-y-3">
            <TypeBreakdown
              byType={stats.by_type}
              heightPx={195}
              animate={!DEMO_MODE}
              selectedType={selectedType}
              onTypeClick={toggleTypeFilter}
            />
            {DEMO_MODE ? (
              <DemoFPRateChart chartHeight={125} />
            ) : (
              <FPRateChart refreshToken={refreshToken} chartHeight={125} />
            )}
            {DEMO_MODE ? (
              <DemoModelHealth
                expanded={healthDetailOpen}
                onExpand={() => setHealthDetailOpen(true)}
                onClose={() => setHealthDetailOpen(false)}
              />
            ) : (
              <ModelHealth
                refreshToken={refreshToken}
                expanded={healthDetailOpen}
                onExpand={() => setHealthDetailOpen(true)}
                onClose={() => setHealthDetailOpen(false)}
              />
            )}
          </div>
        </div>
      </main>

      {DEMO_MODE ? (
        <>
          <AlertDetailDrawer
            alert={selectedAlert}
            onClose={() => setSelectedAlert(null)}
            onSelect={setSelectedAlert}
          />
          <SimulateDrawer
            open={simulateOpen}
            onClose={() => setSimulateOpen(false)}
          />
        </>
      ) : (
        <LiveSimulateDrawer
          open={simulateOpen}
          onClose={() => setSimulateOpen(false)}
        />
      )}
    </div>
  );
}

export default App;
