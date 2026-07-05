/**
 * Tests for the demo-mode StatusBar.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import StatusBar from "../components/StatusBar";
import { clockTime } from "../demo/format";
import { makeTestEngine } from "./helpers";

const T0 = new Date("2024-12-01T06:00:10Z").getTime();

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(T0);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("StatusBar", () => {
  it("shows the LIVE indicator and fleet reporting count", () => {
    const engine = makeTestEngine({ fleetSize: 50 });
    render(<StatusBar onSimulate={() => {}} engine={engine} />);

    expect(screen.getByText("LIVE")).toBeInTheDocument();
    const { vehiclesRecent } = engine.getSnapshot();
    expect(screen.getByText(`${vehiclesRecent}/50`)).toBeInTheDocument();
    expect(screen.getByText(/vehicles reporting/)).toBeInTheDocument();
  });

  it("shows the last-event age", () => {
    render(<StatusBar onSimulate={() => {}} engine={makeTestEngine()} />);
    expect(screen.getByText(/last event (just now|\d+[smh] ago)/)).toBeInTheDocument();
  });

  it("ticks the clock every second", () => {
    render(<StatusBar onSimulate={() => {}} engine={makeTestEngine()} />);
    expect(screen.getByText(clockTime(T0))).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByText(clockTime(T0 + 1000))).toBeInTheDocument();
  });

  it("invokes onSimulate when the Simulate button is clicked", () => {
    const onSimulate = vi.fn();
    render(<StatusBar onSimulate={onSimulate} engine={makeTestEngine()} />);
    fireEvent.click(screen.getByText("Simulate"));
    expect(onSimulate).toHaveBeenCalledTimes(1);
  });

  it("dismisses the synthetic-replay notice", () => {
    render(<StatusBar onSimulate={() => {}} engine={makeTestEngine()} />);
    expect(screen.getByText(/Synthetic replay/)).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Dismiss notice"));
    expect(screen.queryByText(/Synthetic replay/)).not.toBeInTheDocument();
  });
});
