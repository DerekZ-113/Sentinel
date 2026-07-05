/**
 * Demo-mode status bar: LIVE indicator, clock, fleet reporting count,
 * last-event age, Simulate trigger, and a dismissible synthetic-replay
 * notice. Replaces the old DemoBanner.
 */

import { useState } from "react";
import { getEngine } from "../demo/engineInstance";
import { useEngineSnapshot, useNow } from "../demo/useEngine";
import { clockTime, relativeTime } from "../demo/format";
import type { ReplayEngine } from "../demo/types";

interface StatusBarProps {
  onSimulate: () => void;
  engine?: ReplayEngine;
}

export default function StatusBar({ onSimulate, engine = getEngine() }: StatusBarProps) {
  const snapshot = useEngineSnapshot(engine);
  const now = useNow(1000);
  const [noticeDismissed, setNoticeDismissed] = useState(false);

  const lastEvent =
    snapshot.lastEventAt !== null
      ? relativeTime(now, new Date(snapshot.lastEventAt).toISOString())
      : "—";

  return (
    <div className="sticky top-14 md:top-0 z-20 -mx-4 md:-mx-8 px-4 md:px-8 py-2.5 bg-gray-950/90 backdrop-blur-sm border-b border-gray-800">
      <div className="flex items-center gap-x-4 gap-y-1 flex-wrap text-xs">
        <span className="flex items-center gap-1.5 font-semibold text-emerald-400">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60 motion-reduce:hidden" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
          </span>
          LIVE
        </span>

        <span className="text-gray-400 font-mono tabular-nums">{clockTime(now)}</span>

        <span className="text-gray-400">
          <span className="text-white font-medium tabular-nums">
            {snapshot.vehiclesRecent}/{snapshot.vehiclesTotal}
          </span>{" "}
          vehicles reporting
        </span>

        <span className="text-gray-500 tabular-nums">last event {lastEvent}</span>

        <div className="ml-auto flex items-center gap-3">
          {!noticeDismissed && (
            <span className="hidden sm:flex items-center gap-2 text-gray-500 bg-gray-900/80 border border-gray-800 rounded-full px-3 py-1">
              Synthetic replay — not real fleet data.{" "}
              <a
                href="https://github.com/DerekZ-113/Sentinel"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300 underline"
              >
                GitHub
              </a>
              <button
                onClick={() => setNoticeDismissed(true)}
                className="text-gray-500 hover:text-gray-300"
                aria-label="Dismiss notice"
              >
                ✕
              </button>
            </span>
          )}
          <button
            onClick={onSimulate}
            className="bg-blue-600 hover:bg-blue-500 text-white font-medium px-3 py-1.5 rounded-lg transition-colors"
          >
            Simulate
          </button>
        </div>
      </div>
    </div>
  );
}
