/**
 * Tests for the demo mode heuristic predictor.
 */

import { describe, it, expect } from "vitest";
import { demoPredict } from "../services/demoPredict";
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

describe("demoPredict", () => {
  it("stuck + heavy traffic → suppress", () => {
    const result = demoPredict(makePayload({ notification_type: "stuck", traffic_condition: "heavy" }));
    expect(result.needs_intervention).toBe(false);
    expect(result.confidence).toBeGreaterThanOrEqual(0.80);
    expect(result.confidence).toBeLessThanOrEqual(0.92);
  });

  it("stuck + clear road + light → flag", () => {
    const result = demoPredict(makePayload({
      notification_type: "stuck", traffic_condition: "light", construction_zone: "none",
    }));
    expect(result.needs_intervention).toBe(true);
  });

  it("object_query + moving → flag", () => {
    const result = demoPredict(makePayload({
      notification_type: "verification_request",
      notification_subtype: "object_query",
      speed: 15,
    }));
    expect(result.needs_intervention).toBe(true);
    expect(result.confidence).toBeGreaterThanOrEqual(0.85);
  });

  it("object_query + object in path → flag even in high pedestrian area", () => {
    const result = demoPredict(makePayload({
      notification_type: "verification_request",
      notification_subtype: "object_query",
      object_in_path: true,
      speed: 3,
      pedestrian_density: 0.7,
    }));
    expect(result.needs_intervention).toBe(true);
    expect(result.confidence).toBeGreaterThanOrEqual(0.85);
  });

  it("object_query + high pedestrian → suppress", () => {
    const result = demoPredict(makePayload({
      notification_type: "verification_request",
      notification_subtype: "object_query",
      speed: 3,
      pedestrian_density: 0.7,
    }));
    expect(result.needs_intervention).toBe(false);
  });

  it("ev_alert + far → suppress", () => {
    const result = demoPredict(makePayload({
      notification_type: "emergency_vehicle_alert",
      ev_distance: 300,
    }));
    expect(result.needs_intervention).toBe(false);
  });

  it("ev_alert + close → flag", () => {
    const result = demoPredict(makePayload({
      notification_type: "emergency_vehicle_alert",
      ev_distance: 30,
    }));
    expect(result.needs_intervention).toBe(true);
  });

  it("passenger_assist → always flag", () => {
    const result = demoPredict(makePayload({ notification_type: "passenger_assist" }));
    expect(result.needs_intervention).toBe(true);
    expect(result.confidence).toBeGreaterThanOrEqual(0.95);
  });

  it("raw_score matches confidence direction", () => {
    const flagged = demoPredict(makePayload({ notification_type: "passenger_assist" }));
    expect(flagged.raw_score).toBeCloseTo(flagged.confidence, 1);

    const suppressed = demoPredict(makePayload({
      notification_type: "stuck", traffic_condition: "heavy",
    }));
    expect(suppressed.raw_score).toBeCloseTo(1 - suppressed.confidence, 1);
  });

  it("returns vehicle_id and timestamp", () => {
    const result = demoPredict(makePayload({ vehicle_id: "test_123" }));
    expect(result.vehicle_id).toBe("test_123");
    expect(result.timestamp).toBeTruthy();
  });
});
