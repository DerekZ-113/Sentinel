/**
 * Tests for the shared demo-mode decision rule table.
 */

import { describe, it, expect } from "vitest";
import { evaluateRules } from "../services/decisionRules";
import type { NotificationPayload } from "../services/api";

function makePayload(overrides: Partial<NotificationPayload> = {}): NotificationPayload {
  return {
    vehicle_id: "test_v",
    speed: 30,
    expected_speed: 35,
    road_type: "main_road",
    traffic_condition: "moderate",
    construction_zone: "none",
    notification_type: "stuck",
    pedestrian_density: 0.3,
    object_in_path: false,
    time_since_stop: 0,
    ...overrides,
  };
}

describe("evaluateRules", () => {
  it("object_query + object_in_path → flag 0.85-0.95, beating pedestrian suppression (H7)", () => {
    // High pedestrian density would suppress if object_in_path were ignored
    const decision = evaluateRules(makePayload({
      notification_type: "verification_request",
      notification_subtype: "object_query",
      object_in_path: true,
      speed: 3,
      pedestrian_density: 0.7,
    }));
    expect(decision.needsIntervention).toBe(true);
    expect(decision.confidenceRange).toEqual([0.85, 0.95]);
    expect(decision.factors.some((f) => f.label === "Obstruction confirmed")).toBe(true);
  });

  it("object_query without obstruction keeps pedestrian-based verdicts", () => {
    const suppressed = evaluateRules(makePayload({
      notification_type: "verification_request",
      notification_subtype: "object_query",
      object_in_path: false,
      speed: 3,
      pedestrian_density: 0.7,
    }));
    expect(suppressed.needsIntervention).toBe(false);

    const flagged = evaluateRules(makePayload({
      notification_type: "verification_request",
      notification_subtype: "object_query",
      object_in_path: false,
      speed: 3,
      pedestrian_density: 0.1,
    }));
    expect(flagged.needsIntervention).toBe(true);
  });

  it("stuck + heavy traffic → suppress with a suppress-direction factor", () => {
    const decision = evaluateRules(makePayload({
      notification_type: "stuck",
      traffic_condition: "heavy",
    }));
    expect(decision.needsIntervention).toBe(false);
    expect(decision.confidenceRange).toEqual([0.8, 0.92]);
    expect(decision.factors[0].direction).toBe("suppress");
  });

  it("emergency_vehicle_alert with null distance treated as far away → suppress", () => {
    const decision = evaluateRules(makePayload({
      notification_type: "emergency_vehicle_alert",
      ev_distance: null,
    }));
    expect(decision.needsIntervention).toBe(false);
    expect(decision.factors[0].label).toBe("EV far away");
  });

  it("lane_mapping_verify verdict is driven by the injected rng", () => {
    const payload = makePayload({
      notification_type: "verification_request",
      notification_subtype: "lane_mapping_verify",
    });
    expect(evaluateRules(payload, () => 0.9).needsIntervention).toBe(true);
    expect(evaluateRules(payload, () => 0.1).needsIntervention).toBe(false);
  });

  it("every notification type yields at least one factor", () => {
    const types = [
      "stuck",
      "verification_request",
      "emergency_vehicle_alert",
      "speed_anomaly",
      "impact_l0",
      "passenger_assist",
      "something_unknown",
    ];
    for (const t of types) {
      const decision = evaluateRules(makePayload({ notification_type: t }));
      expect(decision.factors.length).toBeGreaterThan(0);
    }
  });

  it("appends a stationary context factor for stopped vehicles", () => {
    const decision = evaluateRules(makePayload({
      notification_type: "stuck",
      speed: 0,
      time_since_stop: 145,
    }));
    const stationary = decision.factors.find((f) => f.label === "Stationary");
    expect(stationary).toBeDefined();
    expect(stationary?.detail).toContain("145");
    expect(stationary?.direction).toBe("context");
  });
});
