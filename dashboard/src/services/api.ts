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

type ValidationIssue = {
  loc?: Array<string | number>;
  msg?: string;
};

function formatValidationIssue(issue: unknown): string {
  if (typeof issue !== "object" || issue === null) {
    return String(issue);
  }

  const validationIssue = issue as ValidationIssue;
  const location = Array.isArray(validationIssue.loc)
    ? validationIssue.loc.join(".")
    : "";
  const message = validationIssue.msg ?? JSON.stringify(issue);

  return location ? `${location}: ${message}` : message;
}

function formatErrorDetail(body: unknown): string | undefined {
  if (typeof body === "string") return body;
  if (typeof body !== "object" || body === null) return undefined;

  const detail = (body as { detail?: unknown; message?: unknown; error?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map(formatValidationIssue).join("; ");
  }

  const message = (body as { message?: unknown }).message;
  if (typeof message === "string") return message;

  const error = (body as { error?: unknown }).error;
  if (typeof error === "string") return error;

  return undefined;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = init === undefined ? await fetch(url) : await fetch(url, init);

  if (res.ok === false) {
    let detail: string | undefined;
    try {
      detail = formatErrorDetail(await res.json());
    } catch {
      detail = undefined;
    }

    const baseMessage = `Request to ${url} failed with ${res.status} ${res.statusText}`;
    throw new Error(detail ? `${baseMessage}: ${detail}` : baseMessage);
  }

  return res.json() as Promise<T>;
}

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
  return fetchJson<HealthResponse>("/health");
}

export async function fetchStats(hours = 24): Promise<StatsResponse> {
  if (DEMO_MODE) return Promise.resolve(demoStats as StatsResponse);
  return fetchJson<StatsResponse>(`${API_BASE}/stats?hours=${hours}`);
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
  return fetchJson<AlertsResponse>(url);
}

export async function fetchModelHealth(
  hours = 24
): Promise<ModelHealthResponse> {
  if (DEMO_MODE)
    return Promise.resolve(demoModelHealth as ModelHealthResponse);
  return fetchJson<ModelHealthResponse>(
    `${API_BASE}/stats/model-health?hours=${hours}`
  );
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
  return fetchJson<PredictionResponse>(`${API_BASE}/predict`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
}

export async function fetchFPOverTime(
  hours = 24,
  buckets = 12
): Promise<FPOverTimeResponse> {
  if (DEMO_MODE)
    return Promise.resolve(demoFPOverTime as FPOverTimeResponse);
  return fetchJson<FPOverTimeResponse>(
    `${API_BASE}/stats/fp-over-time?hours=${hours}&buckets=${buckets}`
  );
}
