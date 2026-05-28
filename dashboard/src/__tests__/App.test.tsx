/**
 * Tests for dashboard live refresh orchestration in App.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import { fetchHealth, fetchStats } from "../services/api";

vi.mock("../services/api", () => ({
  fetchHealth: vi.fn(),
  fetchStats: vi.fn(),
}));

vi.mock("../components/DemoBanner", () => ({
  default: () => null,
}));

vi.mock("../components/OverviewCards", () => ({
  default: ({ stats }: { stats: { total_alerts: number } }) => (
    <div>Total Alerts {stats.total_alerts}</div>
  ),
}));

vi.mock("../components/TypeBreakdown", () => ({
  default: () => <div>Alerts by Type</div>,
}));

vi.mock("../components/SimulatePanel", () => ({
  default: () => <div>Simulate Notification</div>,
}));

vi.mock("../components/AlertFeed", () => ({
  default: ({ refreshToken }: { refreshToken?: number }) => (
    <div data-testid="alert-feed">Recent Alerts token {refreshToken ?? 0}</div>
  ),
}));

vi.mock("../components/FPRateChart", () => ({
  default: ({ refreshToken }: { refreshToken?: number }) => (
    <div data-testid="fp-chart">FP Rate Trend token {refreshToken ?? 0}</div>
  ),
}));

vi.mock("../components/ModelHealth", () => ({
  default: ({ refreshToken }: { refreshToken?: number }) => (
    <div data-testid="model-health">Model Health token {refreshToken ?? 0}</div>
  ),
}));

const fetchHealthMock = vi.mocked(fetchHealth);
const fetchStatsMock = vi.mocked(fetchStats);

const baseHealth = {
  status: "healthy",
  model_loaded: true,
  db_connected: true,
  model_features: 28,
  model_threshold: 0.5,
  uptime_seconds: 10,
};

function stats(totalAlerts = 100) {
  return {
    time_window_hours: 24,
    total_alerts: totalAlerts,
    total_flagged: 30,
    total_suppressed: 70,
    overall_fp_rate: 0.2,
    by_type: [],
  };
}

function mockIntersectionObserver() {
  class MockIntersectionObserver {
    observe = vi.fn();
    unobserve = vi.fn();
    disconnect = vi.fn();
  }

  vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
}

describe("App live refresh", () => {
  const originalDemoMode = import.meta.env.VITE_DEMO_MODE;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    import.meta.env.VITE_DEMO_MODE = "false";
    mockIntersectionObserver();
    fetchHealthMock.mockResolvedValue(baseHealth);
    fetchStatsMock.mockResolvedValue(stats());
  });

  afterEach(() => {
    import.meta.env.VITE_DEMO_MODE = originalDemoMode;
    vi.useRealTimers();
  });

  it("renders the Live Refresh control after initial data loads", async () => {
    render(<App />);

    expect(await screen.findByText("Live Refresh")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Turn live refresh on" })).toHaveTextContent("Off");
    expect(screen.getByText(/Last updated/)).toBeInTheDocument();
  });

  it("runs an immediate refresh when enabled", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("Live Refresh");
    await user.click(screen.getByRole("button", { name: "Turn live refresh on" }));

    await waitFor(() => {
      expect(fetchHealthMock).toHaveBeenCalledTimes(2);
      expect(fetchStatsMock).toHaveBeenCalledTimes(2);
    });
    expect(screen.getByTestId("alert-feed")).toHaveTextContent("token 1");
  });

  it("polls every 5000ms while enabled", async () => {
    render(<App />);

    await screen.findByText("Live Refresh");
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Turn live refresh on" }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchHealthMock).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    expect(fetchHealthMock).toHaveBeenCalledTimes(3);
    expect(fetchStatsMock).toHaveBeenCalledTimes(3);
    expect(screen.getByTestId("model-health")).toHaveTextContent("token 2");
  });

  it("stops polling when disabled", async () => {
    render(<App />);

    await screen.findByText("Live Refresh");
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Turn live refresh on" }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchHealthMock).toHaveBeenCalledTimes(2);

    fireEvent.click(screen.getByRole("button", { name: "Turn live refresh off" }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });

    expect(fetchHealthMock).toHaveBeenCalledTimes(2);
    expect(fetchStatsMock).toHaveBeenCalledTimes(2);
  });

  it("hides Live Refresh in static demo mode and does not poll", async () => {
    const setIntervalSpy = vi.spyOn(window, "setInterval");
    import.meta.env.VITE_DEMO_MODE = "true";
    render(<App />);

    await screen.findByText("Total Alerts 100");
    expect(screen.queryByText("Live Refresh")).not.toBeInTheDocument();
    expect(fetchHealthMock).toHaveBeenCalledTimes(1);
    expect(fetchStatsMock).toHaveBeenCalledTimes(1);
    expect(setIntervalSpy.mock.calls.some(([, delay]) => delay === 5000)).toBe(false);
  });

  it("keeps existing summary data visible when a later refresh fails", async () => {
    const user = userEvent.setup();
    fetchStatsMock
      .mockResolvedValueOnce(stats(100))
      .mockRejectedValueOnce(new Error("Stats unavailable"));
    render(<App />);

    expect(await screen.findByText("Total Alerts 100")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Turn live refresh on" }));

    expect(await screen.findByText("Refresh failed: Stats unavailable")).toBeInTheDocument();
    expect(screen.getByText("Total Alerts 100")).toBeInTheDocument();
  });
});
