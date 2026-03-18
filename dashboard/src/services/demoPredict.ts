/**
 * Demo mode heuristic predictor.
 *
 * Domain-rule-based prediction that produces plausible results
 * matching the patterns the XGBoost model learned.
 */

import type { NotificationPayload, PredictionResponse } from "./api";

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function makeResult(
  payload: NotificationPayload,
  needsIntervention: boolean,
  confidenceMin: number,
  confidenceMax: number,
): PredictionResponse {
  const confidence = rand(confidenceMin, confidenceMax);
  return {
    vehicle_id: payload.vehicle_id,
    notification_type: payload.notification_type,
    needs_intervention: needsIntervention,
    confidence: Math.round(confidence * 1000) / 1000,
    raw_score: Math.round(
      (needsIntervention ? confidence : 1 - confidence) * 1000
    ) / 1000,
    timestamp: new Date().toISOString(),
  };
}

export function demoPredict(payload: NotificationPayload): PredictionResponse {
  const { notification_type, notification_subtype, traffic_condition, construction_zone, speed, pedestrian_density, ev_distance, road_type } = payload;

  // Stuck notifications
  if (notification_type === "stuck") {
    if (traffic_condition === "heavy" || traffic_condition === "standstill") {
      return makeResult(payload, false, 0.80, 0.92);
    }
    if (construction_zone && construction_zone !== "none") {
      return makeResult(payload, false, 0.75, 0.88);
    }
    if (traffic_condition === "light") {
      return makeResult(payload, true, 0.70, 0.85);
    }
    return makeResult(payload, false, 0.60, 0.75);
  }

  // Verification requests
  if (notification_type === "verification_request") {
    if (notification_subtype === "object_query") {
      if (speed > 10) return makeResult(payload, true, 0.85, 0.95);
      if (pedestrian_density > 0.5) return makeResult(payload, false, 0.80, 0.92);
      if (pedestrian_density <= 0.3) return makeResult(payload, true, 0.75, 0.88);
      return makeResult(payload, false, 0.65, 0.80);
    }
    if (notification_subtype === "traffic_signal_verify") {
      return makeResult(payload, true, 0.70, 0.80);
    }
    if (notification_subtype === "lane_mapping_verify") {
      return makeResult(payload, Math.random() > 0.7, 0.65, 0.78);
    }
    return makeResult(payload, true, 0.60, 0.75);
  }

  // Emergency vehicle alerts
  if (notification_type === "emergency_vehicle_alert") {
    const dist = ev_distance ?? 999;
    if (dist > 200) return makeResult(payload, false, 0.85, 0.95);
    if (dist < 50) return makeResult(payload, true, 0.80, 0.92);
    return makeResult(payload, false, 0.60, 0.75);
  }

  // Speed anomaly
  if (notification_type === "speed_anomaly") {
    if (traffic_condition === "heavy" || traffic_condition === "standstill") {
      return makeResult(payload, false, 0.80, 0.90);
    }
    if (traffic_condition === "light") {
      return makeResult(payload, true, 0.75, 0.85);
    }
    return makeResult(payload, false, 0.65, 0.80);
  }

  // Impact
  if (notification_type === "impact_l0") {
    if (road_type === "residential" || road_type === "downtown") {
      return makeResult(payload, false, 0.60, 0.75);
    }
    return makeResult(payload, true, 0.70, 0.85);
  }

  // Passenger assist — always flag
  if (notification_type === "passenger_assist") {
    return makeResult(payload, true, 0.95, 0.99);
  }

  // Fallback
  return makeResult(payload, true, 0.50, 0.70);
}
