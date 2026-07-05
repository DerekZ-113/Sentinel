/**
 * Shared fixtures for demo-mode component tests.
 *
 * Tests inject a small deterministic engine via props — they never touch
 * VITE_DEMO_MODE or the app-wide singleton.
 */

import { createReplayEngine } from "../demo/replayEngine";
import { mulberry32 } from "../demo/random";
import type { AlertRecord } from "../services/api";
import type { EngineOptions, ReplayEngine } from "../demo/types";

export function makePool(n: number): AlertRecord[] {
  return Array.from({ length: n }, (_, i) => ({
    id: i + 1,
    time: "2024-12-01T00:00:00Z",
    vehicle_id: `vehicle_${(i % 5).toString().padStart(3, "0")}`,
    notification_type: i % 2 === 0 ? "stuck" : "speed_anomaly",
    notification_subtype: null,
    needs_intervention_predicted: i % 3 === 0,
    needs_intervention_actual: i % 4 === 0 ? null : i % 3 === 0,
    confidence: 0.5 + (i % 5) * 0.1,
    speed: 10,
    road_type: "downtown",
    traffic_condition: "heavy",
  }));
}

/** Deterministic engine: seeded rng, clock via Date.now (fake-timer friendly). */
export function makeTestEngine(overrides: Partial<EngineOptions> = {}): ReplayEngine {
  return createReplayEngine({
    pool: makePool(20),
    rng: mulberry32(42),
    now: () => Date.now(),
    ...overrides,
  });
}

/** Fixed 1-deal-per-second tempo with no bursts. */
export const STEADY_TEMPO = {
  meanGapMs: 1000,
  minGapMs: 1000,
  maxGapMs: 1000,
  burstProbability: 0,
};
