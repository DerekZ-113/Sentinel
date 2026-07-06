/**
 * Demo-mode alert stream: newest-on-top slice of the replay engine's ring.
 * Pauses while hovered or scrolled away from the top, collecting a
 * "{n} new alerts" pill instead of yanking rows out from under the cursor.
 */

import { useRef, useState } from "react";
import type { UIEvent } from "react";
import { getEngine } from "../demo/engineInstance";
import { useEngineSnapshot, useNow } from "../demo/useEngine";
import { relativeTime } from "../demo/format";
import type { DemoAlert, ReplayEngine } from "../demo/types";
import { ConfidenceBar, TypeBadge } from "./alertRowParts";

const FEED_SIZE = 50;

interface LiveAlertFeedProps {
  engine?: ReplayEngine;
  onSelect?: (alert: DemoAlert) => void;
  /** Constant for a mount — App remounts the feed via key when it changes. */
  filterType?: string | null;
}

export default function LiveAlertFeed({
  engine = getEngine(),
  onSelect,
  filterType = null,
}: LiveAlertFeedProps) {
  const snapshot = useEngineSnapshot(engine);
  const now = useNow(1000);
  const [frozen, setFrozen] = useState<DemoAlert[] | null>(null);
  const [hovering, setHovering] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  // Rows newer than the first paint slide in; the initial backlog doesn't
  const [initialNewestId] = useState(() => snapshot.events[0]?.id ?? 0);

  const matchesFilter = (alert: DemoAlert) =>
    filterType === null || alert.notification_type === filterType;
  const filteredEvents = snapshot.events.filter(matchesFilter);

  const visible = frozen ?? filteredEvents.slice(0, FEED_SIZE);
  // Count-based rather than id arithmetic: under a filter, ids in the
  // visible stream are no longer consecutive.
  const pending =
    frozen !== null
      ? filteredEvents.filter((e) => e.id > (frozen[0]?.id ?? 0)).length
      : 0;

  function freeze() {
    setFrozen(
      (current) =>
        current ??
        engine.getSnapshot().events.filter(matchesFilter).slice(0, FEED_SIZE)
    );
  }

  function resume() {
    setFrozen(null);
    scrollRef.current?.scrollTo(0, 0);
  }

  function handleMouseEnter() {
    setHovering(true);
    freeze();
  }

  function handleMouseLeave() {
    setHovering(false);
    if ((scrollRef.current?.scrollTop ?? 0) <= 4) setFrozen(null);
  }

  function handleScroll(event: UIEvent<HTMLDivElement>) {
    if (event.currentTarget.scrollTop > 4) {
      freeze();
    } else if (!hovering) {
      setFrozen(null);
    }
  }

  return (
    <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-5 overflow-hidden max-h-[420px] lg:max-h-none lg:h-full min-h-0 flex flex-col relative">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Alert Stream</h2>
        <span className="text-xs text-gray-500 tabular-nums">
          {snapshot.totalDealt.toLocaleString()} this shift
        </span>
      </div>

      {pending > 0 && (
        <button
          onClick={resume}
          className="absolute left-1/2 -translate-x-1/2 top-14 z-10 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium px-3 py-1 rounded-full shadow-lg transition-colors"
        >
          {pending} new {pending === 1 ? "alert" : "alerts"} ↑
        </button>
      )}

      <div
        ref={scrollRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onScroll={handleScroll}
        className="overflow-auto flex-1"
      >
        <table className="w-full text-sm table-fixed">
          <thead>
            <tr className="text-gray-400 text-xs uppercase">
              <th className="sticky top-0 z-10 bg-gray-900 border-b border-gray-700/50 text-left py-2 pr-3 w-16">
                Time
              </th>
              <th className="sticky top-0 z-10 bg-gray-900 border-b border-gray-700/50 text-left py-2 pr-3 w-32">
                Vehicle
              </th>
              <th className="sticky top-0 z-10 bg-gray-900 border-b border-gray-700/50 text-left py-2 pr-3">
                Type
              </th>
              <th className="sticky top-0 z-10 bg-gray-900 border-b border-gray-700/50 text-left py-2 pr-3 w-24">
                Prediction
              </th>
              <th className="sticky top-0 z-10 bg-gray-900 border-b border-gray-700/50 text-right py-2 w-20">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody>
            {visible.map((alert) => (
              <tr
                key={alert.id}
                onClick={() => onSelect?.(alert)}
                className={`border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors ${
                  onSelect ? "cursor-pointer" : ""
                } ${alert.id > initialNewestId ? "alert-row-in" : ""}`}
              >
                <td className="py-2.5 pr-3 text-gray-500 text-xs whitespace-nowrap tabular-nums">
                  {relativeTime(now, alert.time)}
                </td>
                <td className="py-2.5 pr-3 text-gray-300 font-mono text-xs whitespace-nowrap">
                  {alert.vehicle_id}
                  {alert.source === "manual" && (
                    <span className="ml-1.5 inline-block px-1.5 py-0.5 rounded text-[10px] bg-blue-900/60 text-blue-300 border border-blue-800/50 font-sans">
                      manual
                    </span>
                  )}
                </td>
                <td className="py-2.5 pr-3">
                  <TypeBadge
                    type={alert.notification_type}
                    subtype={alert.notification_subtype}
                  />
                </td>
                <td className="py-2.5 pr-3 whitespace-nowrap">
                  {alert.needs_intervention_predicted ? (
                    <span className="text-red-400 font-medium">⚠ Flag</span>
                  ) : (
                    <span className="text-emerald-400 font-medium">✓ Suppress</span>
                  )}
                </td>
                <td className="py-2.5 text-right font-mono text-xs">
                  <ConfidenceBar value={alert.confidence} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
