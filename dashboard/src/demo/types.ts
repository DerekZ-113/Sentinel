/**
 * Types for the demo-mode replay engine.
 */

import type {
  AlertRecord,
  FPBucket,
  ModelHealthResponse,
  StatsResponse,
} from "../services/api";

/** A dealt alert. Manual alerts come from the Simulate drawer. */
export type DemoAlert = AlertRecord & { source?: "replay" | "manual" };

/**
 * Immutable snapshot of everything the engine derives from the dealt stream.
 * All dashboard surfaces read from here — single source of truth.
 */
export interface EngineSnapshot {
  /** Ring buffer of dealt alerts, newest first, capped at ringCap. */
  events: DemoAlert[];
  /** Cumulative alerts dealt since shift start (survives ring eviction). */
  totalDealt: number;
  /** Cumulative stats since shift start (StatsResponse shape). */
  stats: StatsResponse;
  /** Cumulative model health since shift start (ModelHealthResponse shape). */
  modelHealth: ModelHealthResponse;
  /** Rolling wall-clock-aligned FP buckets over the recent window. */
  fpBuckets: FPBucket[];
  /** Distinct vehicles seen in the recent-activity window. */
  vehiclesRecent: number;
  /** Fleet size ("N of vehiclesTotal reporting"). */
  vehiclesTotal: number;
  /** Epoch ms of the newest dealt event, null before any event. */
  lastEventAt: number | null;
  /** Epoch ms of the oldest pre-warmed event ("shift start"). */
  shiftStartMs: number;
}

export interface EngineTempo {
  /** Mean gap between dealt alerts (exponential distribution). */
  meanGapMs: number;
  minGapMs: number;
  maxGapMs: number;
  /** Probability that a deal is a burst (1-2 extra alerts at once). */
  burstProbability: number;
}

export interface EngineOptions {
  /** Source alerts replayed on a shuffled loop. Must be non-empty. */
  pool: AlertRecord[];
  /** Injectable randomness for deterministic tests. Defaults to Math.random. */
  rng?: () => number;
  /** Injectable clock for deterministic tests. Defaults to Date.now. */
  now?: () => number;
  tempo?: Partial<EngineTempo>;
  /** Ring buffer capacity. Default 2000. */
  ringCap?: number;
  /** How much history to pre-deal at creation. Default: the full pool. */
  prewarmCount?: number;
  /** Fleet size reported in the snapshot. Default 50. */
  fleetSize?: number;
}

export interface ReplayEngine {
  /** Start dealing. Idempotent while running (StrictMode-safe). */
  start(): void;
  /** Stop timers. The snapshot stays readable; start() resumes. */
  stop(): void;
  subscribe(listener: () => void): () => void;
  getSnapshot(): EngineSnapshot;
  /** Push a Simulate-drawer prediction into the live stream. */
  injectManual(alert: Omit<DemoAlert, "id" | "source">): DemoAlert;
}
