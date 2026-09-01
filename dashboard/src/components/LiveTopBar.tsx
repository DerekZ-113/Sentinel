/**
 * Live-mode top bar: composes the shared TopBar shell with the live
 * refresh toggle, last-updated stamp, refresh errors, and the Simulate
 * drawer trigger. All data arrives via props from App.
 */

import TopBar from "./TopBar";
import type { HealthResponse } from "../services/api";

interface LiveTopBarProps {
  health: Pick<HealthResponse, "status" | "model_features">;
  liveRefreshEnabled: boolean;
  onToggleLiveRefresh: () => void;
  lastUpdatedText: string | null;
  refreshError: string | null;
  onSimulate: () => void;
}

export default function LiveTopBar({
  health,
  liveRefreshEnabled,
  onToggleLiveRefresh,
  lastUpdatedText,
  refreshError,
  onSimulate,
}: LiveTopBarProps) {
  return (
    <TopBar health={health}>
      <div className="flex items-center gap-2">
        <span className="text-ink-micro text-[10px] uppercase tracking-[0.1em]">
          Live Refresh
        </span>
        <button
          type="button"
          onClick={onToggleLiveRefresh}
          aria-label={
            liveRefreshEnabled ? "Turn live refresh off" : "Turn live refresh on"
          }
          aria-pressed={liveRefreshEnabled}
          className={`rounded-xs px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.08em] transition-colors ${
            liveRefreshEnabled
              ? "bg-ok/10 text-ok border border-ok/50"
              : "bg-inset text-ink-mid border border-hairline-2"
          }`}
        >
          {liveRefreshEnabled ? "On" : "Off"}
        </button>
      </div>

      {lastUpdatedText && (
        <p className="text-ink-low text-[10px]">Last updated {lastUpdatedText}</p>
      )}
      {liveRefreshEnabled && refreshError && (
        <p className="text-warn text-[10px] leading-snug">
          {refreshError}
        </p>
      )}

      <div className="ml-auto">
        <button
          onClick={onSimulate}
          className="border border-accent/60 text-accent hover:bg-accent/10 rounded-xs px-3 py-1.5 text-[11px] font-medium uppercase tracking-[0.1em] transition-colors"
        >
          Simulate
        </button>
      </div>
    </TopBar>
  );
}
