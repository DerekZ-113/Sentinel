/**
 * Lazy app-wide replay engine singleton.
 *
 * Demo-only module: it is only reachable through demo components selected by
 * module-scope DEMO_MODE ternaries, so live builds tree-shake it (and the
 * fixture import) away. start() is idempotent, which makes the singleton
 * safe under React StrictMode double-mounting.
 */

import demoAlerts from "../data/alerts.json";
import type { AlertsResponse } from "../services/api";
import { createReplayEngine } from "./replayEngine";
import type { ReplayEngine } from "./types";

const DEMO_ALERTS: AlertsResponse = demoAlerts;

let engine: ReplayEngine | null = null;

export function getEngine(): ReplayEngine {
  if (engine === null) {
    engine = createReplayEngine({ pool: DEMO_ALERTS.alerts });
    engine.start();
  }
  return engine;
}
