/**
 * Tests for the demo-mode replay engine.
 *
 * Deterministic: seeded rng (mulberry32) + vitest fake timers. The engine
 * reads time through an injected `() => Date.now()`, which fake timers
 * advance in lockstep with setTimeout/setInterval.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createReplayEngine } from "../demo/replayEngine";
import { mulberry32 } from "../demo/random";
import {
  aggregatesToModelHealth,
  aggregatesToStats,
  applyAlert,
  emptyAggregates,
  FP_BUCKET_MS,
} from "../demo/derive";
import type { AlertRecord } from "../services/api";
import type { EngineOptions } from "../demo/types";

// Epoch ms divisible by FP_BUCKET_MS, so T0 sits 10s into a bucket
const T0 = new Date("2024-12-01T06:00:10Z").getTime();

function makePool(n: number): AlertRecord[] {
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

function makeEngine(overrides: Partial<EngineOptions> = {}) {
  return createReplayEngine({
    pool: makePool(20),
    rng: mulberry32(42),
    now: () => Date.now(),
    ...overrides,
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(T0);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("createReplayEngine", () => {
  it("rejects an empty pool", () => {
    expect(() => makeEngine({ pool: [] })).toThrow(/non-empty/);
  });

  it("pre-warms the full pool before start(): counters seeded, newest first", () => {
    const engine = makeEngine();
    const snap = engine.getSnapshot();

    expect(snap.events).toHaveLength(20);
    expect(snap.totalDealt).toBe(20);
    expect(snap.stats.total_alerts).toBe(20);
    // Newest first, all in the past, shift start = oldest event
    const times = snap.events.map((e) => Date.parse(e.time));
    for (let i = 1; i < times.length; i++) {
      expect(times[i - 1]).toBeGreaterThanOrEqual(times[i]);
    }
    expect(times[0]).toBeLessThanOrEqual(T0);
    expect(snap.shiftStartMs).toBe(times[times.length - 1]);
    // ids ascend with time (oldest = 1)
    expect(snap.events[snap.events.length - 1].id).toBe(1);
    expect(snap.events[0].id).toBe(20);
  });

  it("deals live events after start(), stamped with the current clock", () => {
    const engine = makeEngine();
    engine.start();
    const before = engine.getSnapshot().totalDealt;

    vi.advanceTimersByTime(60_000);

    const snap = engine.getSnapshot();
    expect(snap.totalDealt).toBeGreaterThan(before);
    const newest = snap.events[0];
    expect(Date.parse(newest.time)).toBeGreaterThan(T0);
    expect(Date.parse(newest.time)).toBeLessThanOrEqual(Date.now());
    expect(newest.source).toBe("replay");
    expect(snap.lastEventAt).toBe(Date.parse(newest.time));
    engine.stop();
  });

  it("is deterministic: same seed and clock produce identical streams", () => {
    const a = makeEngine({ rng: mulberry32(7) });
    const b = makeEngine({ rng: mulberry32(7) });
    a.start();
    b.start();
    vi.advanceTimersByTime(120_000);

    const snapA = a.getSnapshot();
    const snapB = b.getSnapshot();
    expect(snapA.totalDealt).toBe(snapB.totalDealt);
    expect(snapA.events.map((e) => `${e.vehicle_id}@${e.time}`)).toEqual(
      snapB.events.map((e) => `${e.vehicle_id}@${e.time}`)
    );
    a.stop();
    b.stop();
  });

  it("single source of truth: incremental aggregates equal brute-force recompute", () => {
    const engine = makeEngine();
    engine.start();
    vi.advanceTimersByTime(90_000);

    const snap = engine.getSnapshot();
    // Within ring cap, snapshot.events is the complete dealt history
    expect(snap.events).toHaveLength(snap.totalDealt);

    const brute = emptyAggregates();
    for (const event of snap.events) applyAlert(brute, event);
    expect(aggregatesToStats(brute, snap.stats.time_window_hours)).toEqual(snap.stats);
    expect(aggregatesToModelHealth(brute)).toEqual(snap.modelHealth);
    engine.stop();
  });

  it("aligns FP buckets to wall-clock boundaries and counts new deals into the last bucket", () => {
    const engine = makeEngine({
      // Slow tempo: exactly one deal at +5s, nothing else for a while
      tempo: { meanGapMs: 5000, minGapMs: 5000, maxGapMs: 5000, burstProbability: 0 },
    });
    const initial = engine.getSnapshot().fpBuckets;
    expect(initial).toHaveLength(12);
    for (const bucket of initial) {
      expect(Date.parse(bucket.time) % FP_BUCKET_MS).toBe(0);
    }
    for (let i = 1; i < initial.length; i++) {
      expect(Date.parse(initial[i].time) - Date.parse(initial[i - 1].time)).toBe(FP_BUCKET_MS);
    }
    const lastStart = Date.parse(initial[11].time);
    expect(lastStart).toBeLessThanOrEqual(T0);
    expect(T0).toBeLessThan(lastStart + FP_BUCKET_MS);

    engine.start();
    const lastTotalBefore = initial[11].total;
    vi.advanceTimersByTime(5_000);
    const after = engine.getSnapshot().fpBuckets;
    expect(after[11].total).toBe(lastTotalBefore + 1);
    engine.stop();
  });

  it("rolls the bucket window forward when the clock crosses a boundary", () => {
    const engine = makeEngine({
      tempo: { meanGapMs: 600_000, minGapMs: 600_000, maxGapMs: 600_000, burstProbability: 0 },
    });
    engine.start();
    const before = engine.getSnapshot().fpBuckets;

    vi.advanceTimersByTime(FP_BUCKET_MS + 1000);

    const after = engine.getSnapshot().fpBuckets;
    expect(Date.parse(after[11].time) - Date.parse(before[11].time)).toBe(FP_BUCKET_MS);
    engine.stop();
  });

  it("injectManual pushes a manual alert into stream and aggregates", () => {
    const engine = makeEngine();
    const statsBefore = engine.getSnapshot().stats;

    const manual = engine.injectManual({
      time: new Date(T0).toISOString(),
      vehicle_id: "sim_abcd",
      notification_type: "passenger_assist",
      notification_subtype: null,
      needs_intervention_predicted: true,
      needs_intervention_actual: null,
      confidence: 0.97,
      speed: 0,
      road_type: "downtown",
      traffic_condition: "heavy",
    });

    const snap = engine.getSnapshot();
    expect(manual.source).toBe("manual");
    expect(snap.events[0]).toBe(manual);
    expect(snap.totalDealt).toBe(statsBefore.total_alerts + 1);
    expect(snap.stats.total_flagged).toBe(statsBefore.total_flagged + 1);
    const paType = snap.stats.by_type.find((t) => t.notification_type === "passenger_assist");
    expect(paType?.total).toBe(1);
    // Synthetic sim vehicles must not inflate the fleet-reporting count
    expect(snap.vehiclesRecent).toBeLessThanOrEqual(5);
  });

  it("evicts oldest events past ringCap while cumulative counters keep counting", () => {
    const engine = makeEngine({ ringCap: 10 });
    const snap = engine.getSnapshot();
    expect(snap.events).toHaveLength(10);
    expect(snap.totalDealt).toBe(20);
    expect(snap.stats.total_alerts).toBe(20);
    // Newest 10 survive (ids 11..20)
    expect(snap.events[0].id).toBe(20);
    expect(snap.events[9].id).toBe(11);
  });

  it("keeps snapshot identity stable across idle heartbeats", () => {
    const engine = makeEngine({
      // Prewarm gaps ~100s and no deal for 100s — nothing changes in 3s
      tempo: { meanGapMs: 100_000, minGapMs: 100_000, maxGapMs: 100_000, burstProbability: 0 },
      prewarmCount: 3,
      pool: makePool(5),
    });
    engine.start();
    const snap1 = engine.getSnapshot();

    vi.advanceTimersByTime(3_000); // three heartbeats, no boundary crossing

    expect(engine.getSnapshot()).toBe(snap1);
    engine.stop();
  });

  it("stop() halts dealing; start() resumes", () => {
    const engine = makeEngine();
    engine.start();
    vi.advanceTimersByTime(30_000);
    engine.stop();
    const stoppedAt = engine.getSnapshot().totalDealt;

    vi.advanceTimersByTime(60_000);
    expect(engine.getSnapshot().totalDealt).toBe(stoppedAt);

    engine.start();
    vi.advanceTimersByTime(30_000);
    expect(engine.getSnapshot().totalDealt).toBeGreaterThan(stoppedAt);
    engine.stop();
  });

  it("cycles the pool with a reshuffle instead of running dry", () => {
    const pool = makePool(3);
    const engine = makeEngine({ pool, prewarmCount: 3 });
    engine.start();
    vi.advanceTimersByTime(120_000); // far more deals than pool size

    const snap = engine.getSnapshot();
    expect(snap.totalDealt).toBeGreaterThan(10);
    const poolVehicles = new Set(pool.map((p) => p.vehicle_id));
    for (const event of snap.events) {
      expect(poolVehicles.has(event.vehicle_id)).toBe(true);
    }
    engine.stop();
  });

  it("notifies subscribers on deals and supports unsubscribe", () => {
    const engine = makeEngine();
    engine.start();
    const listener = vi.fn();
    const unsubscribe = engine.subscribe(listener);

    vi.advanceTimersByTime(30_000);
    expect(listener).toHaveBeenCalled();

    const callsAtUnsubscribe = listener.mock.calls.length;
    unsubscribe();
    vi.advanceTimersByTime(30_000);
    expect(listener.mock.calls.length).toBe(callsAtUnsubscribe);
    engine.stop();
  });
});
