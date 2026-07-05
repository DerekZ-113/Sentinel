/**
 * Tests for the demo-mode SimulateDrawer, including injection of
 * predictions into the replay stream.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SimulateDrawer from "../components/SimulateDrawer";
import { buildAlertFromPrediction } from "../demo/replayEngine";
import { makeTestEngine } from "./helpers";
import type { NotificationPayload, PredictionResponse } from "../services/api";

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("SimulateDrawer", () => {
  it("renders nothing when closed", () => {
    const { container } = render(
      <SimulateDrawer open={false} onClose={() => {}} engine={makeTestEngine()} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the simulate form when open and closes via the button", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<SimulateDrawer open onClose={onClose} engine={makeTestEngine()} />);

    expect(screen.getByText("Simulate Notification")).toBeInTheDocument();
    await user.click(screen.getByLabelText("Close simulate"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("injects a manual alert into the engine on successful prediction", async () => {
    const user = userEvent.setup();
    const engine = makeTestEngine();
    const before = engine.getSnapshot().totalDealt;

    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      json: () =>
        Promise.resolve({
          vehicle_id: "sim_test",
          notification_type: "stuck",
          needs_intervention: true,
          confidence: 0.88,
          raw_score: 0.88,
          timestamp: "2024-12-01T06:01:00Z",
        }),
    } as Response);

    render(<SimulateDrawer open onClose={() => {}} engine={engine} />);
    await user.click(screen.getByText("Run Prediction"));

    await waitFor(() => {
      expect(engine.getSnapshot().totalDealt).toBe(before + 1);
    });
    const newest = engine.getSnapshot().events[0];
    expect(newest.source).toBe("manual");
    expect(newest.vehicle_id).toBe("sim_test");
    expect(newest.needs_intervention_predicted).toBe(true);
  });
});

describe("buildAlertFromPrediction", () => {
  it("maps prediction and payload fields onto an alert record", () => {
    const result: PredictionResponse = {
      vehicle_id: "sim_abcd",
      notification_type: "verification_request",
      needs_intervention: true,
      confidence: 0.9,
      raw_score: 0.9,
      timestamp: "2024-12-01T06:02:00Z",
    };
    const payload: NotificationPayload = {
      vehicle_id: "sim_abcd",
      speed: 12,
      expected_speed: 25,
      road_type: "downtown",
      traffic_condition: "light",
      construction_zone: "none",
      notification_type: "verification_request",
      notification_subtype: "object_query",
      ev_distance: null,
      pedestrian_density: 0.2,
      object_in_path: true,
      time_since_stop: 0,
    };

    const alert = buildAlertFromPrediction(result, payload);
    expect(alert.time).toBe("2024-12-01T06:02:00Z");
    expect(alert.vehicle_id).toBe("sim_abcd");
    expect(alert.notification_subtype).toBe("object_query");
    expect(alert.needs_intervention_predicted).toBe(true);
    // Manual alerts have no ground truth or coordinates
    expect(alert.needs_intervention_actual).toBeNull();
    expect(alert.latitude).toBeNull();
    expect(alert.longitude).toBeNull();
    expect(alert.object_in_path).toBe(true);
    expect(alert.raw_score).toBe(0.9);
  });
});
