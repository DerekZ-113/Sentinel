/**
 * Demo-mode bootstrap: same shape as useLiveBootstrap, but health comes from
 * the static fixture and stats stream from the replay engine.
 *
 * Demo-only module — tree-shaken out of live builds via the module-scope
 * DEMO_MODE ternary in App.tsx.
 */

import demoHealth from "../data/health.json";
import type { HealthResponse } from "../services/api";
import type { LiveBootstrap } from "../hooks/useLiveBootstrap";
import { getEngine } from "./engineInstance";
import { useEngineSnapshot } from "./useEngine";

const DEMO_HEALTH: HealthResponse = demoHealth;

function noop(): void {}

export function useDemoBootstrap(): LiveBootstrap {
  const snapshot = useEngineSnapshot(getEngine());
  return {
    health: DEMO_HEALTH,
    stats: snapshot.stats,
    error: null,
    refreshError: null,
    lastUpdatedAt: null,
    retry: noop,
    refresh: noop,
  };
}
