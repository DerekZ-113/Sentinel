/**
 * Tests for the demo-mode AlertDetailDrawer.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import AlertDetailDrawer from "../components/AlertDetailDrawer";
import type { DemoAlert } from "../demo/types";
import { makeTestEngine } from "./helpers";

const T0 = new Date("2024-12-01T06:00:10Z").getTime();

function makeAlert(overrides: Partial<DemoAlert> = {}): DemoAlert {
  return {
    id: 9001,
    time: new Date(T0 - 30_000).toISOString(),
    vehicle_id: "vehicle_042",
    notification_type: "verification_request",
    notification_subtype: "object_query",
    needs_intervention_predicted: true,
    needs_intervention_actual: true,
    confidence: 0.91,
    speed: 3,
    road_type: "downtown",
    traffic_condition: "moderate",
    expected_speed: 25,
    construction_zone: "none",
    pedestrian_density: 0.7,
    ev_distance: null,
    object_in_path: true,
    time_since_stop: 45,
    raw_score: 0.91,
    latitude: 37.55,
    longitude: -122.25,
    ...overrides,
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(T0);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("AlertDetailDrawer", () => {
  it("renders nothing when no alert is selected", () => {
    const { container } = render(
      <AlertDetailDrawer
        alert={null}
        onClose={() => {}}
        onSelect={() => {}}
        engine={makeTestEngine()}
      />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows vehicle, verdict, ground truth, and model score", () => {
    render(
      <AlertDetailDrawer
        alert={makeAlert()}
        onClose={() => {}}
        onSelect={() => {}}
        engine={makeTestEngine()}
      />
    );
    expect(screen.getByText("vehicle_042")).toBeInTheDocument();
    expect(screen.getByText("⚠ Flagged for review")).toBeInTheDocument();
    expect(screen.getByText(/Ground truth: real/)).toBeInTheDocument();
    expect(screen.getByText(/correct/)).toBeInTheDocument();
    expect(screen.getByText("0.910")).toBeInTheDocument();
    expect(screen.getByText(/threshold 0.5/)).toBeInTheDocument();
  });

  it("surfaces the obstruction factor for object_in_path alerts (H7)", () => {
    render(
      <AlertDetailDrawer
        alert={makeAlert()}
        onClose={() => {}}
        onSelect={() => {}}
        engine={makeTestEngine()}
      />
    );
    expect(screen.getByText("Obstruction confirmed")).toBeInTheDocument();
  });

  it("renders the telemetry grid and sector map position", () => {
    render(
      <AlertDetailDrawer
        alert={makeAlert()}
        onClose={() => {}}
        onSelect={() => {}}
        engine={makeTestEngine()}
      />
    );
    expect(screen.getByText("3 mph")).toBeInTheDocument();
    expect(screen.getByText("25 mph")).toBeInTheDocument();
    expect(screen.getByText("downtown")).toBeInTheDocument();
    // lat 37.55 / lon -122.25 → middle of the box → row index 2 (C), col index 4 (5)
    expect(screen.getByText("Sector C5")).toBeInTheDocument();
  });

  it("falls back to a hashed sector when coordinates are missing", () => {
    render(
      <AlertDetailDrawer
        alert={makeAlert({ latitude: null, longitude: null })}
        onClose={() => {}}
        onSelect={() => {}}
        engine={makeTestEngine()}
      />
    );
    expect(screen.getByText(/Sector [A-E][1-8]/)).toBeInTheDocument();
  });

  it("lists the vehicle's other alerts from the ring and navigates on click", () => {
    const engine = makeTestEngine();
    // Pool vehicles cycle vehicle_000..vehicle_004 — pick one with history
    const target = engine.getSnapshot().events.find((e) => e.vehicle_id === "vehicle_001");
    const onSelect = vi.fn();
    render(
      <AlertDetailDrawer
        alert={target as DemoAlert}
        onClose={() => {}}
        onSelect={onSelect}
        engine={engine}
      />
    );

    const historyHeading = screen.getByText(/Recent activity/);
    expect(historyHeading).toBeInTheDocument();
    // 20-alert pool over 5 vehicles → 4 events each → 3 others listed
    const historyButtons = screen
      .getAllByRole("button")
      .filter((b) => b.textContent?.includes("ago") || b.textContent?.includes("just now"));
    expect(historyButtons.length).toBeGreaterThan(0);

    fireEvent.click(historyButtons[0]);
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0].vehicle_id).toBe("vehicle_001");
  });

  it("closes via the close button, backdrop, and Escape", () => {
    const onClose = vi.fn();
    render(
      <AlertDetailDrawer
        alert={makeAlert()}
        onClose={onClose}
        onSelect={() => {}}
        engine={makeTestEngine()}
      />
    );

    fireEvent.click(screen.getByLabelText("Close detail"));
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
