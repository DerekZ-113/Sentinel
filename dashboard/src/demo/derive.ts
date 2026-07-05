/**
 * Pure derivation functions for the replay engine.
 *
 * Cumulative surfaces (stats, model health) are maintained incrementally via
 * applyAlert(); windowed surfaces (FP buckets, recent vehicles) are recomputed
 * from the event ring. The aggregation semantics mirror
 * scripts/export_demo_data.py so demo numbers stay consistent with fixtures.
 */

import type {
  FPBucket,
  ModelHealthResponse,
  StatsResponse,
  TypeStats,
} from "../services/api";
import type { DemoAlert } from "./types";

interface GroupCounters {
  total: number;
  flagged: number;
  withTruth: number;
  correct: number;
  flaggedWithTruth: number;
  falsePositives: number;
}

export interface CumulativeAggregates {
  overall: GroupCounters;
  sumConfidence: number;
  confHigh: number;
  confMedium: number;
  confLow: number;
  byType: Map<string, GroupCounters>;
}

function emptyGroup(): GroupCounters {
  return {
    total: 0,
    flagged: 0,
    withTruth: 0,
    correct: 0,
    flaggedWithTruth: 0,
    falsePositives: 0,
  };
}

export function emptyAggregates(): CumulativeAggregates {
  return {
    overall: emptyGroup(),
    sumConfidence: 0,
    confHigh: 0,
    confMedium: 0,
    confLow: 0,
    byType: new Map(),
  };
}

function applyToGroup(group: GroupCounters, alert: DemoAlert): void {
  group.total += 1;
  if (alert.needs_intervention_predicted) group.flagged += 1;
  if (alert.needs_intervention_actual !== null) {
    group.withTruth += 1;
    if (alert.needs_intervention_predicted === alert.needs_intervention_actual) {
      group.correct += 1;
    }
    if (alert.needs_intervention_predicted) {
      group.flaggedWithTruth += 1;
      if (!alert.needs_intervention_actual) group.falsePositives += 1;
    }
  }
}

export function applyAlert(agg: CumulativeAggregates, alert: DemoAlert): void {
  applyToGroup(agg.overall, alert);
  agg.sumConfidence += alert.confidence;
  if (alert.confidence >= 0.9) agg.confHigh += 1;
  else if (alert.confidence >= 0.7) agg.confMedium += 1;
  else agg.confLow += 1;

  let typeGroup = agg.byType.get(alert.notification_type);
  if (!typeGroup) {
    typeGroup = emptyGroup();
    agg.byType.set(alert.notification_type, typeGroup);
  }
  applyToGroup(typeGroup, alert);
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function fpRate(group: GroupCounters): number | null {
  if (group.flaggedWithTruth === 0) return null;
  return round(group.falsePositives / group.flaggedWithTruth, 4);
}

function accuracy(group: GroupCounters): number | null {
  if (group.withTruth === 0) return null;
  return round(group.correct / group.withTruth, 4);
}

export function aggregatesToStats(
  agg: CumulativeAggregates,
  timeWindowHours: number
): StatsResponse {
  const byType: TypeStats[] = [...agg.byType.entries()]
    .sort((a, b) => b[1].total - a[1].total)
    .map(([notificationType, group]) => ({
      notification_type: notificationType,
      total: group.total,
      flagged: group.flagged,
      suppressed: group.total - group.flagged,
      fp_rate: fpRate(group),
      accuracy: accuracy(group),
    }));

  return {
    time_window_hours: timeWindowHours,
    total_alerts: agg.overall.total,
    total_flagged: agg.overall.flagged,
    total_suppressed: agg.overall.total - agg.overall.flagged,
    overall_fp_rate: fpRate(agg.overall),
    by_type: byType,
  };
}

export function aggregatesToModelHealth(
  agg: CumulativeAggregates
): ModelHealthResponse {
  const total = agg.overall.total;
  const avgConfidence = total > 0 ? round(agg.sumConfidence / total, 4) : null;
  const acc = accuracy(agg.overall);

  let status = "healthy";
  if (acc !== null && acc < 0.6) status = "degraded";
  else if (avgConfidence !== null && avgConfidence < 0.7) status = "warning";

  const flaggedByType: Record<string, number> = {};
  const suppressedByType: Record<string, number> = {};
  for (const [type, group] of agg.byType) {
    if (group.flagged > 0) flaggedByType[type] = group.flagged;
    if (group.total - group.flagged > 0) {
      suppressedByType[type] = group.total - group.flagged;
    }
  }

  return {
    status,
    total_predictions: total,
    pct_flagged: total > 0 ? round((agg.overall.flagged / total) * 100, 1) : 0,
    pct_suppressed:
      total > 0 ? round(((total - agg.overall.flagged) / total) * 100, 1) : 0,
    avg_confidence: avgConfidence,
    accuracy: acc,
    confidence_buckets: {
      high: agg.confHigh,
      medium: agg.confMedium,
      low: agg.confLow,
    },
    flagged_by_type: flaggedByType,
    suppressed_by_type: suppressedByType,
  };
}

export const FP_BUCKET_MS = 2.5 * 60_000;
export const FP_BUCKET_COUNT = 12;

/**
 * Rolling FP buckets over the recent window, aligned to wall-clock
 * FP_BUCKET_MS boundaries. The last bucket is the current (partial) one.
 */
export function computeFPBuckets(
  events: readonly DemoAlert[],
  nowMs: number,
  bucketMs: number = FP_BUCKET_MS,
  bucketCount: number = FP_BUCKET_COUNT
): FPBucket[] {
  const lastBucketStart = Math.floor(nowMs / bucketMs) * bucketMs;
  const windowStart = lastBucketStart - (bucketCount - 1) * bucketMs;

  const groups: GroupCounters[] = Array.from({ length: bucketCount }, emptyGroup);

  for (const alert of events) {
    const t = Date.parse(alert.time);
    if (t < windowStart || t >= lastBucketStart + bucketMs) continue;
    const idx = Math.floor((t - windowStart) / bucketMs);
    applyToGroup(groups[idx], alert);
  }

  return groups.map((group, i) => ({
    time: new Date(windowStart + i * bucketMs).toISOString(),
    total: group.total,
    flagged: group.flagged,
    suppressed: group.total - group.flagged,
    fp_rate: fpRate(group),
    accuracy: accuracy(group),
  }));
}

export function bucketsEqual(a: readonly FPBucket[], b: readonly FPBucket[]): boolean {
  if (a.length !== b.length) return false;
  return a.every((bucket, i) => {
    const other = b[i];
    return (
      bucket.time === other.time &&
      bucket.total === other.total &&
      bucket.flagged === other.flagged &&
      bucket.suppressed === other.suppressed &&
      bucket.fp_rate === other.fp_rate &&
      bucket.accuracy === other.accuracy
    );
  });
}

export const RECENT_VEHICLES_WINDOW_MS = 10 * 60_000;

/**
 * Distinct fleet vehicles with an event in the recent-activity window.
 * Manual Simulate injections use synthetic vehicle ids and are excluded,
 * so the count never exceeds the fleet size.
 */
export function countRecentVehicles(
  events: readonly DemoAlert[],
  nowMs: number,
  windowMs: number = RECENT_VEHICLES_WINDOW_MS
): number {
  const seen = new Set<string>();
  for (const alert of events) {
    if (alert.source === "manual") continue;
    if (Date.parse(alert.time) >= nowMs - windowMs) seen.add(alert.vehicle_id);
  }
  return seen.size;
}
