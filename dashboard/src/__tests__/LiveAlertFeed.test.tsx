/**
 * Tests for the demo-mode LiveAlertFeed.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import LiveAlertFeed from "../components/LiveAlertFeed";
import { makeTestEngine, STEADY_TEMPO } from "./helpers";

const T0 = new Date("2024-12-01T06:00:10Z").getTime();

function steadyEngine() {
  return makeTestEngine({ tempo: STEADY_TEMPO });
}

function scrollContainer(): HTMLElement {
  return screen.getByRole("table").parentElement as HTMLElement;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(T0);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("LiveAlertFeed", () => {
  it("renders the prewarmed backlog with the shift counter", () => {
    render(<LiveAlertFeed engine={steadyEngine()} />);
    expect(screen.getByText("Alert Stream")).toBeInTheDocument();
    expect(screen.getByText("20 this shift")).toBeInTheDocument();
    // 20 data rows + 1 header row
    expect(screen.getAllByRole("row")).toHaveLength(21);
  });

  it("streams new alerts in as time passes", () => {
    const engine = steadyEngine();
    engine.start();
    render(<LiveAlertFeed engine={engine} />);

    act(() => {
      vi.advanceTimersByTime(3000);
    });

    expect(screen.getByText("23 this shift")).toBeInTheDocument();
    engine.stop();
  });

  it("pauses on hover and collects a pending pill instead of moving rows", () => {
    const engine = steadyEngine();
    engine.start();
    render(<LiveAlertFeed engine={engine} />);

    fireEvent.mouseEnter(scrollContainer());
    act(() => {
      vi.advanceTimersByTime(3000);
    });

    expect(screen.getByText("3 new alerts ↑")).toBeInTheDocument();
    // Frozen: still showing the 20 prewarmed rows despite 23 dealt
    expect(screen.getAllByRole("row")).toHaveLength(21);
    engine.stop();
  });

  it("resumes when the pending pill is clicked", () => {
    const engine = steadyEngine();
    engine.start();
    render(<LiveAlertFeed engine={engine} />);

    fireEvent.mouseEnter(scrollContainer());
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    fireEvent.click(screen.getByText("2 new alerts ↑"));

    expect(screen.queryByText(/new alert/)).not.toBeInTheDocument();
    expect(screen.getAllByRole("row")).toHaveLength(23);
    engine.stop();
  });

  it("resumes on mouse leave", () => {
    const engine = steadyEngine();
    engine.start();
    render(<LiveAlertFeed engine={engine} />);

    fireEvent.mouseEnter(scrollContainer());
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByText(/new alerts/)).toBeInTheDocument();

    fireEvent.mouseLeave(scrollContainer());
    expect(screen.queryByText(/new alert/)).not.toBeInTheDocument();
    engine.stop();
  });

  it("shows only alerts of the filtered type", () => {
    // makePool alternates stuck / speed_anomaly, so 10 of 20 match
    render(<LiveAlertFeed engine={steadyEngine()} filterType="stuck" />);
    expect(screen.getAllByRole("row")).toHaveLength(11);
  });

  it("counts only matching alerts in the pending pill while frozen", () => {
    const engine = steadyEngine();
    render(<LiveAlertFeed engine={engine} filterType="stuck" />);

    fireEvent.mouseEnter(scrollContainer());
    const manualBase = {
      time: new Date(T0).toISOString(),
      notification_subtype: null,
      needs_intervention_predicted: false,
      needs_intervention_actual: null,
      confidence: 0.9,
      speed: 10,
      road_type: "downtown",
      traffic_condition: "heavy",
    };
    act(() => {
      engine.injectManual({
        ...manualBase,
        vehicle_id: "sim_a",
        notification_type: "stuck",
      });
      engine.injectManual({
        ...manualBase,
        vehicle_id: "sim_b",
        notification_type: "speed_anomaly",
      });
      engine.injectManual({
        ...manualBase,
        vehicle_id: "sim_c",
        notification_type: "stuck",
      });
    });

    expect(screen.getByText("2 new alerts ↑")).toBeInTheDocument();
  });

  it("badges manual alerts and reports row clicks via onSelect", () => {
    const engine = steadyEngine();
    const manual = engine.injectManual({
      time: new Date(T0).toISOString(),
      vehicle_id: "sim_test",
      notification_type: "passenger_assist",
      notification_subtype: null,
      needs_intervention_predicted: true,
      needs_intervention_actual: null,
      confidence: 0.97,
      speed: 0,
      road_type: "downtown",
      traffic_condition: "heavy",
    });

    const onSelect = vi.fn();
    render(<LiveAlertFeed engine={engine} onSelect={onSelect} />);

    expect(screen.getByText("manual")).toBeInTheDocument();
    fireEvent.click(screen.getByText("sim_test"));
    expect(onSelect).toHaveBeenCalledWith(manual);
  });
});
