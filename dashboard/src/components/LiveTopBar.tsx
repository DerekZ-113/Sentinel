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
        <span className="text-gray-500 text-[11px]">Live Refresh</span>
        <button
          type="button"
          onClick={onToggleLiveRefresh}
          aria-label={
            liveRefreshEnabled ? "Turn live refresh off" : "Turn live refresh on"
          }
          aria-pressed={liveRefreshEnabled}
          className={`rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors ${
            liveRefreshEnabled
              ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
              : "bg-gray-800 text-gray-400 border border-gray-700"
          }`}
        >
          {liveRefreshEnabled ? "On" : "Off"}
        </button>
      </div>

      {lastUpdatedText && (
        <p className="text-gray-600 text-[10px]">Last updated {lastUpdatedText}</p>
      )}
      {liveRefreshEnabled && refreshError && (
        <p className="text-yellow-300/80 text-[10px] leading-snug">
          {refreshError}
        </p>
      )}

      <div className="ml-auto">
        <button
          onClick={onSimulate}
          className="bg-blue-600 hover:bg-blue-500 text-white font-medium px-3 py-1.5 rounded-lg transition-colors"
        >
          Simulate
        </button>
      </div>
    </TopBar>
  );
}
