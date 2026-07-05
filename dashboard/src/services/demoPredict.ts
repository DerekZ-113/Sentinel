/**
 * Demo mode heuristic predictor.
 *
 * Thin wrapper over the shared rule table in decisionRules.ts that shapes
 * the decision into a PredictionResponse, matching the patterns the
 * XGBoost model learned.
 */

import type { NotificationPayload, PredictionResponse } from "./api";
import { evaluateRules } from "./decisionRules";

function rand(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

export function demoPredict(payload: NotificationPayload): PredictionResponse {
  const decision = evaluateRules(payload);
  const [confidenceMin, confidenceMax] = decision.confidenceRange;
  const confidence = rand(confidenceMin, confidenceMax);

  return {
    vehicle_id: payload.vehicle_id,
    notification_type: payload.notification_type,
    needs_intervention: decision.needsIntervention,
    confidence: Math.round(confidence * 1000) / 1000,
    raw_score:
      Math.round(
        (decision.needsIntervention ? confidence : 1 - confidence) * 1000
      ) / 1000,
    timestamp: new Date().toISOString(),
  };
}
