/**
 * Sentinel API Service
 * All backend calls go through here.
 *
 * When VITE_DEMO_MODE=true, serves pre-generated data from bundled JSON files
 * instead of hitting the API. The full-stack API mode is completely unaffected.
 */

import demoAlerts from "../data/alerts.json";
import demoStats from "../data/stats.json";
import demoModelHealth from "../data/model-health.json";
import demoHealth from "../data/health.json";
import demoFPOverTime from "../data/fp-over-time.json";
import { demoPredict } from "./demoPredict";

const API_BASE = "/api";
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";

// ============================================================================
// INTERFACES
// ============================================================================

export interface PredictionResponse {
  vehicle_id: string;
  notification_type: string;
  needs_intervention: boolean;
  confidence: number;
  raw_score: number;
  timestamp: string;
}

export interface AlertRecord {
  id: number;
  time: string;
  vehicle_id: string;
  notification_type: string;
  notification_subtype: string | null;
  needs_intervention_predicted: boolean;
  needs_intervention_actual: boolean | null;
  confidence: number;
  speed: number | null;
  road_type: string | null;
  traffic_condition: string | null;
}

export interface AlertsResponse {
  alerts: AlertRecord[];
  total: number;
  limit: number;
  offset: number;
}

export interface TypeStats {
  notification_type: string;
  total: number;
  flagged: number;
  suppressed: number;
  fp_rate: number | null;
  accuracy: number | null;
}

export interface StatsResponse {
  time_window_hours: number;
  total_alerts: number;
  total_flagged: number;
  total_suppressed: number;
  overall_fp_rate: number | null;
  by_type: TypeStats[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  db_connected: boolean;
  model_features: number;
  model_threshold: number;
  uptime_seconds: number;
}

export interface NotificationPayload {
  vehicle_id: string;
  speed: number;
  expected_speed: number;
  road_type: string;
  traffic_condition: string;
  construction_zone: string;
  notification_type: string;
  notification_subtype?: string | null;
  ev_distance?: number | null;
  pedestrian_density: number;
  object_in_path: boolean;
  time_since_stop: number;
  hour_of_day?: number;
}

export interface ModelHealthResponse {
  status: string;
  total_predictions: number;
  pct_flagged: number;
  pct_suppressed: number;
  avg_confidence: number | null;
  accuracy: number | null;
  confidence_buckets: { high: number; medium: number; low: number };
  flagged_by_type: Record<string, number>;
  suppressed_by_type: Record<string, number>;
}

export interface FPBucket {
  time: string;
  total: number;
  flagged: number;
  suppressed: number;
  fp_rate: number | null;
  accuracy: number | null;
}

export interface FPOverTimeResponse {
  time_window_hours: number;
  buckets: FPBucket[];
}

// ============================================================================
// API CALLS
// ============================================================================

export async function fetchHealth(): Promise<HealthResponse> {
  if (DEMO_MODE) return Promise.resolve(demoHealth as HealthResponse);
  const res = await fetch("/health");
  return res.json();
}

export async function fetchStats(hours = 24): Promise<StatsResponse> {
  if (DEMO_MODE) return Promise.resolve(demoStats as StatsResponse);
  const res = await fetch(`${API_BASE}/stats?hours=${hours}`);
  return res.json();
}

export async function fetchAlerts(
  limit = 50,
  offset = 0,
  notificationType?: string
): Promise<AlertsResponse> {
  if (DEMO_MODE) {
    let alerts = (demoAlerts as AlertsResponse).alerts;
    if (notificationType) {
      alerts = alerts.filter((a) => a.notification_type === notificationType);
    }
    // Already sorted by time desc in the JSON
    const total = alerts.length;
    const paged = alerts.slice(offset, offset + limit);
    return Promise.resolve({ alerts: paged, total, limit, offset });
  }
  let url = `${API_BASE}/alerts?limit=${limit}&offset=${offset}`;
  if (notificationType) {
    url += `&notification_type=${notificationType}`;
  }
  const res = await fetch(url);
  return res.json();
}

export async function fetchModelHealth(
  hours = 24
): Promise<ModelHealthResponse> {
  if (DEMO_MODE)
    return Promise.resolve(demoModelHealth as ModelHealthResponse);
  const res = await fetch(`${API_BASE}/stats/model-health?hours=${hours}`);
  return res.json();
}

export async function postPredict(
  payload: NotificationPayload
): Promise<PredictionResponse> {
  if (DEMO_MODE) return Promise.resolve(demoPredict(payload));
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const apiKey = import.meta.env.VITE_API_KEY;
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function fetchFPOverTime(
  hours = 24,
  buckets = 12
): Promise<FPOverTimeResponse> {
  if (DEMO_MODE)
    return Promise.resolve(demoFPOverTime as FPOverTimeResponse);
  const res = await fetch(
    `${API_BASE}/stats/fp-over-time?hours=${hours}&buckets=${buckets}`
  );
  return res.json();
}
