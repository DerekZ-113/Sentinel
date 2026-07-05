/**
 * Tests for FPRateChart component.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import FPRateChart from "../components/FPRateChart";

const mockData = {
  time_window_hours: 24,
  buckets: [
    { time: "2024-12-01T06:00:00Z", total: 80, flagged: 10, suppressed: 70, fp_rate: 0.0, accuracy: 0.85 },
    { time: "2024-12-01T07:00:00Z", total: 90, flagged: 12, suppressed: 78, fp_rate: 0.08, accuracy: 0.82 },
    { time: "2024-12-01T08:00:00Z", total: 95, flagged: 15, suppressed: 80, fp_rate: 0.0, accuracy: 0.80 },
  ],
};

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("FPRateChart", () => {
  it("shows loading state initially", () => {
    vi.spyOn(globalThis, "fetch").mockReturnValue(new Promise(() => {}));
    render(<FPRateChart />);
    expect(screen.getByText(/Loading FP rate trend/)).toBeInTheDocument();
  });

  it("renders chart after data loads", async () => {
    vi.spyOn(globalThis, "fetch").mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve(mockData),
      } as Response)
    );
    render(<FPRateChart />);
    await waitFor(() => {
      expect(screen.getByText("FP Rate Over Time")).toBeInTheDocument();
    });
  });

  it("labels the window from time_window_hours, not bucket count", async () => {
    vi.spyOn(globalThis, "fetch").mockReturnValue(
      Promise.resolve({
        json: () => Promise.resolve(mockData),
      } as Response)
    );
    render(<FPRateChart />);
    await waitFor(() => {
      // mockData has 3 buckets over a 24h window — the label must say 24
      expect(screen.getByText("Last 24 hours")).toBeInTheDocument();
    });
  });

  it("shows error state on fetch failure", async () => {
    vi.spyOn(globalThis, "fetch").mockReturnValue(
      Promise.reject(new Error("Network error"))
    );
    render(<FPRateChart />);
    await waitFor(() => {
      expect(screen.getByText("Network error")).toBeInTheDocument();
      expect(screen.getByText("Retry")).toBeInTheDocument();
    });
  });

  it("refetches when refreshToken changes", async () => {
    // The header labels the window from time_window_hours (D5), so the
    // refreshed payload widens the window to make the refetch observable
    const refreshedData = {
      ...mockData,
      time_window_hours: 48,
      buckets: [
        ...mockData.buckets,
        { time: "2024-12-01T09:00:00Z", total: 105, flagged: 20, suppressed: 85, fp_rate: 0.05, accuracy: 0.9 },
      ],
    };
    const spy = vi.spyOn(globalThis, "fetch")
      .mockReturnValueOnce(
        Promise.resolve({
          json: () => Promise.resolve(mockData),
        } as Response)
      )
      .mockReturnValueOnce(
        Promise.resolve({
          json: () => Promise.resolve(refreshedData),
        } as Response)
      );

    const { rerender } = render(<FPRateChart refreshToken={0} />);
    await waitFor(() => {
      expect(screen.getByText("Last 24 hours")).toBeInTheDocument();
    });

    rerender(<FPRateChart refreshToken={1} />);

    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(2);
      expect(screen.getByText("Last 48 hours")).toBeInTheDocument();
    });
  });
});
