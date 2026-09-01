/**
 * Demo-mode status bar: LIVE indicator, clock, fleet reporting count,
 * last-event age, Simulate trigger, and the always-visible SIM
 * annunciator — rendered on the shared TopBar shell.
 */

import TopBar from "./TopBar";
import { getEngine } from "../demo/engineInstance";
import { useEngineSnapshot, useNow } from "../demo/useEngine";
import { clockTime, relativeTime } from "../demo/format";
import type { ReplayEngine } from "../demo/types";
import type { HealthResponse } from "../services/api";

interface StatusBarProps {
  health: Pick<HealthResponse, "status" | "model_features">;
  onSimulate: () => void;
  engine?: ReplayEngine;
}

export default function StatusBar({
  health,
  onSimulate,
  engine = getEngine(),
}: StatusBarProps) {
  const snapshot = useEngineSnapshot(engine);
  const now = useNow(1000);

  const lastEvent =
    snapshot.lastEventAt !== null
      ? relativeTime(now, new Date(snapshot.lastEventAt).toISOString())
      : "—";

  return (
    <TopBar health={health}>
      <span className="flex items-center gap-1.5 font-semibold tracking-[0.08em] text-ok">
        <span className="h-2 w-2 bg-ok" />
        LIVE
      </span>

      <span className="text-ink-data tabular-nums">{clockTime(now) + "Z"}</span>

      <span className="text-[10px] uppercase tracking-[0.1em] text-ink-micro">
        <span className="text-ink font-medium tabular-nums">
          {snapshot.vehiclesRecent}/{snapshot.vehiclesTotal}
        </span>{" "}
        vehicles reporting
      </span>

      <span className="text-[10px] uppercase tracking-[0.1em] text-ink-low tabular-nums">
        last event {lastEvent}
      </span>

      <div className="ml-auto flex items-center gap-3">
        <span className="hidden sm:flex items-center gap-2 border border-warn/50 bg-warn/10 text-warn rounded-xs px-2.5 py-0.5 text-[10px] uppercase tracking-[0.1em]">
          SIM · Synthetic replay
          <a
            href="https://github.com/DerekZ-113/Sentinel"
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent hover:text-ink underline underline-offset-2"
          >
            GitHub
          </a>
        </span>
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
