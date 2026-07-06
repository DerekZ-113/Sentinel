/**
 * Tests for dashboard live refresh orchestration in App.
 *
 * DEMO_MODE is a module-scope constant (required for tree-shaking), so the
 * demo-mode test re-imports App via vi.resetModules() with the env set,
 * instead of flipping import.meta.env at runtime.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App";
import { fetchHealth, fetchStats } from "../services/api";

vi.mock("../services/api", () => ({
  fetchHealth: vi.fn(),
  fetchStats: vi.fn(),
  // Getter: the mock instance is cached across vi.resetModules(), but each
  // re-imported App reads this property fresh — false at initial file load,
  // true for the demo test's resetModules + dynamic import
  get DEMO_MODE() {
    return import.meta.env.VITE_DEMO_MODE === "true";
  },
}));

vi.mock("../demo/demoData", () => ({
  useDemoBootstrap: () => ({
    health: {
      status: "healthy",
      model_loaded: true,
      db_connected: true,
      model_features: 28,
      model_threshold: 0.5,
      uptime_seconds: 10,
    },
    stats: {
      time_window_hours: 24,
      total_alerts: 100,
      total_flagged: 30,
      total_suppressed: 70,
      overall_fp_rate: 0.2,
      by_type: [],
    },
    error: null,
    refreshError: null,
    lastUpdatedAt: null,
    retry: () => {},
    refresh: () => {},
  }),
}));

vi.mock("../components/OverviewCards", () => ({
  default: ({ stats }: { stats: { total_alerts: number } }) => (
    <div>Total Alerts {stats.total_alerts}</div>
  ),
}));

vi.mock("../components/TypeBreakdown", () => ({
  default: ({ onTypeClick }: { onTypeClick?: (t: string) => void }) => (
    <div>
      Alerts by Type
      <button onClick={() => onTypeClick?.("stuck")}>select stuck</button>
    </div>
  ),
}));

vi.mock("../components/AlertFeed", () => ({
  default: ({
    refreshToken,
    filterType,
  }: {
    refreshToken?: number;
    filterType?: string | null;
  }) => (
    <div data-testid="alert-feed">
      Recent Alerts token {refreshToken ?? 0} filter {filterType ?? "none"}
    </div>
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

vi.mock("../components/StatusBar", () => ({
  default: () => <div data-testid="status-bar">LIVE</div>,
}));

vi.mock("../components/LiveAlertFeed", () => ({
  default: ({ filterType }: { filterType?: string | null }) => (
    <div data-testid="live-feed">Alert Stream filter {filterType ?? "none"}</div>
  ),
}));

vi.mock("../components/demo/DemoFPRateChart", () => ({
  default: () => <div data-testid="demo-fp-chart">FP Rate Trend</div>,
}));

vi.mock("../components/demo/DemoModelHealth", () => ({
  default: () => <div data-testid="demo-model-health">Model Health</div>,
}));

vi.mock("../components/AlertDetailDrawer", () => ({
  default: () => null,
}));

vi.mock("../components/SimulateDrawer", () => ({
  default: () => null,
}));

vi.mock("../components/LiveSimulateDrawer", () => ({
  default: ({ open }: { open: boolean }) =>
    open ? <div data-testid="live-simulate-drawer">Simulate Drawer</div> : null,
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

describe("App live refresh", () => {
  const originalDemoMode = import.meta.env.VITE_DEMO_MODE;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    import.meta.env.VITE_DEMO_MODE = "false";
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
    vi.resetModules();
    const { default: DemoApp } = await import("../App");
    render(<DemoApp />);

    expect(await screen.findByText("Total Alerts 100")).toBeInTheDocument();
    expect(screen.getByTestId("status-bar")).toBeInTheDocument();
    expect(screen.getByTestId("live-feed")).toBeInTheDocument();
    expect(screen.queryByText("Live Refresh")).not.toBeInTheDocument();
    // Demo mode never touches the API
    expect(fetchHealthMock).toHaveBeenCalledTimes(0);
    expect(fetchStatsMock).toHaveBeenCalledTimes(0);
    expect(setIntervalSpy.mock.calls.some(([, delay]) => delay === 5000)).toBe(false);
  });

  it("opens the simulate drawer from the top bar", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("Live Refresh");
    expect(screen.queryByTestId("live-simulate-drawer")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Simulate" }));
    expect(screen.getByTestId("live-simulate-drawer")).toBeInTheDocument();
  });

  it("filters the feed from the type breakdown and clears via the chip", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("Live Refresh");
    expect(screen.getByTestId("alert-feed")).toHaveTextContent("filter none");

    await user.click(screen.getByRole("button", { name: "select stuck" }));
    expect(screen.getByTestId("alert-feed")).toHaveTextContent("filter stuck");

    await user.click(screen.getByRole("button", { name: "Clear type filter" }));
    expect(screen.getByTestId("alert-feed")).toHaveTextContent("filter none");
    expect(
      screen.queryByRole("button", { name: "Clear type filter" })
    ).not.toBeInTheDocument();
  });

  it("toggles the filter off when the same type is clicked again", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("Live Refresh");
    await user.click(screen.getByRole("button", { name: "select stuck" }));
    expect(screen.getByTestId("alert-feed")).toHaveTextContent("filter stuck");

    await user.click(screen.getByRole("button", { name: "select stuck" }));
    expect(screen.getByTestId("alert-feed")).toHaveTextContent("filter none");
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
