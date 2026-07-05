/**
 * Demo-mode replay engine.
 *
 * Deals the bundled alert pool onto a live timeline: a seeded pre-warm
 * history at creation, then Poisson-scheduled live deals with occasional
 * bursts. Every dashboard surface derives from the snapshot this engine
 * publishes — nothing else fabricates numbers.
 *
 * Pure TS with injectable rng/now; timers are plain setTimeout/setInterval
 * so tests can drive it with fake timers.
 */

import { expGap, shuffle } from "./random";
import type { Rng } from "./random";
import {
  aggregatesToModelHealth,
  aggregatesToStats,
  applyAlert,
  bucketsEqual,
  computeFPBuckets,
  countRecentVehicles,
  emptyAggregates,
} from "./derive";
import type {
  DemoAlert,
  EngineOptions,
  EngineSnapshot,
  EngineTempo,
  ReplayEngine,
} from "./types";
import type {
  AlertRecord,
  FPBucket,
  NotificationPayload,
  PredictionResponse,
} from "../services/api";

export const DEFAULT_TEMPO: EngineTempo = {
  meanGapMs: 4000,
  minGapMs: 600,
  maxGapMs: 12000,
  burstProbability: 0.12,
};

const DEFAULT_RING_CAP = 2000;
const DEFAULT_FLEET_SIZE = 50;
const HEARTBEAT_MS = 1000;

/** Shape a Simulate-drawer prediction into an injectable alert record. */
export function buildAlertFromPrediction(
  result: PredictionResponse,
  payload: NotificationPayload
): Omit<AlertRecord, "id"> {
  return {
    time: result.timestamp,
    vehicle_id: result.vehicle_id,
    notification_type: result.notification_type,
    notification_subtype: payload.notification_subtype ?? null,
    needs_intervention_predicted: result.needs_intervention,
    needs_intervention_actual: null,
    confidence: result.confidence,
    speed: payload.speed,
    road_type: payload.road_type,
    traffic_condition: payload.traffic_condition,
    expected_speed: payload.expected_speed,
    construction_zone: payload.construction_zone,
    pedestrian_density: payload.pedestrian_density,
    ev_distance: payload.ev_distance ?? null,
    object_in_path: payload.object_in_path,
    time_since_stop: payload.time_since_stop,
    raw_score: result.raw_score,
    latitude: null,
    longitude: null,
  };
}

export function createReplayEngine(options: EngineOptions): ReplayEngine {
  if (options.pool.length === 0) {
    throw new Error("replay engine requires a non-empty alert pool");
  }

  const pool = options.pool;
  const rng: Rng = options.rng ?? Math.random;
  const now = options.now ?? Date.now;
  const tempo: EngineTempo = { ...DEFAULT_TEMPO, ...options.tempo };
  const ringCap = options.ringCap ?? DEFAULT_RING_CAP;
  const prewarmCount = options.prewarmCount ?? pool.length;
  const fleetSize = options.fleetSize ?? DEFAULT_FLEET_SIZE;

  let order: AlertRecord[] = shuffle(pool, rng);
  let poolIdx = 0;
  let nextId = 1;

  let events: DemoAlert[] = [];
  const agg = emptyAggregates();
  let fpBuckets: FPBucket[] = [];
  let vehiclesRecent = 0;
  let lastEventAt: number | null = null;
  let shiftStartMs = now();

  let running = false;
  let dealTimer: ReturnType<typeof setTimeout> | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  const listeners = new Set<() => void>();
  let snapshot: EngineSnapshot;

  function nextFromPool(): AlertRecord {
    if (poolIdx >= order.length) {
      order = shuffle(pool, rng);
      poolIdx = 0;
    }
    return order[poolIdx++];
  }

  function deal(timeMs: number): DemoAlert {
    return {
      ...nextFromPool(),
      id: nextId++,
      time: new Date(timeMs).toISOString(),
      source: "replay",
    };
  }

  function pushEvent(alert: DemoAlert): void {
    events = [alert, ...events];
    if (events.length > ringCap) events = events.slice(0, ringCap);
    applyAlert(agg, alert);
    const t = Date.parse(alert.time);
    if (lastEventAt === null || t > lastEventAt) lastEventAt = t;
  }

  /** Recompute windowed surfaces; returns true when their content changed. */
  function refreshWindowed(): boolean {
    const nowMs = now();
    const nextBuckets = computeFPBuckets(events, nowMs);
    const nextVehicles = countRecentVehicles(events, nowMs);

    let changed = false;
    if (!bucketsEqual(fpBuckets, nextBuckets)) {
      fpBuckets = nextBuckets;
      changed = true;
    }
    if (nextVehicles !== vehiclesRecent) {
      vehiclesRecent = nextVehicles;
      changed = true;
    }
    return changed;
  }

  function rebuildSnapshot(): void {
    const hours = Math.max(1, Math.round((now() - shiftStartMs) / 3_600_000));
    snapshot = {
      events,
      totalDealt: agg.overall.total,
      stats: aggregatesToStats(agg, hours),
      modelHealth: aggregatesToModelHealth(agg),
      fpBuckets,
      vehiclesRecent,
      vehiclesTotal: fleetSize,
      lastEventAt,
      shiftStartMs,
    };
  }

  function notify(): void {
    for (const listener of listeners) listener();
  }

  function prewarm(): void {
    const startMs = now();
    const times: number[] = [];
    let t = startMs;
    for (let i = 0; i < prewarmCount; i++) {
      t -= expGap(rng, tempo.meanGapMs, tempo.minGapMs, tempo.maxGapMs);
      // Floor to whole ms so shiftStartMs matches the truncated ISO timestamps
      times.push(Math.floor(t));
    }
    times.reverse(); // oldest first, so ids ascend with time
    shiftStartMs = times[0] ?? startMs;
    for (const timeMs of times) pushEvent(deal(timeMs));
    refreshWindowed();
    rebuildSnapshot();
  }

  function dealLive(): void {
    pushEvent(deal(now()));
  }

  function scheduleNext(): void {
    const gap = expGap(rng, tempo.meanGapMs, tempo.minGapMs, tempo.maxGapMs);
    dealTimer = setTimeout(fire, gap);
  }

  function fire(): void {
    dealLive();
    if (rng() < tempo.burstProbability) {
      const extras = 1 + Math.floor(rng() * 2);
      for (let i = 0; i < extras; i++) dealLive();
    }
    refreshWindowed();
    rebuildSnapshot();
    notify();
    scheduleNext();
  }

  function onHeartbeat(): void {
    if (refreshWindowed()) {
      rebuildSnapshot();
      notify();
    }
  }

  prewarm();

  return {
    start(): void {
      if (running) return;
      running = true;
      heartbeatTimer = setInterval(onHeartbeat, HEARTBEAT_MS);
      scheduleNext();
    },

    stop(): void {
      running = false;
      if (dealTimer !== null) clearTimeout(dealTimer);
      if (heartbeatTimer !== null) clearInterval(heartbeatTimer);
      dealTimer = null;
      heartbeatTimer = null;
    },

    subscribe(listener: () => void): () => void {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },

    getSnapshot(): EngineSnapshot {
      return snapshot;
    },

    injectManual(alert: Omit<DemoAlert, "id" | "source">): DemoAlert {
      const manual: DemoAlert = { ...alert, id: nextId++, source: "manual" };
      pushEvent(manual);
      refreshWindowed();
      rebuildSnapshot();
      notify();
      return manual;
    },
  };
}
