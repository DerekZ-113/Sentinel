/**
 * React bindings for the replay engine.
 */

import { useEffect, useState, useSyncExternalStore } from "react";
import type { EngineSnapshot, ReplayEngine } from "./types";

export function useEngineSnapshot(engine: ReplayEngine): EngineSnapshot {
  return useSyncExternalStore(engine.subscribe, engine.getSnapshot);
}

/** Wall-clock ticker for relative-time labels ("4s ago") and the clock. */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}
