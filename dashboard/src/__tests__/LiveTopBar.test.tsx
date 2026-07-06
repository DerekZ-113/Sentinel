/**
 * Tests for the live-mode top bar (presentational; all data via props).
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { ComponentProps } from "react";
import LiveTopBar from "../components/LiveTopBar";

type Props = ComponentProps<typeof LiveTopBar>;

function renderBar(overrides: Partial<Props> = {}) {
  const props: Props = {
    health: { status: "healthy", model_features: 28 },
    liveRefreshEnabled: false,
    onToggleLiveRefresh: vi.fn(),
    lastUpdatedText: null,
    refreshError: null,
    onSimulate: vi.fn(),
    ...overrides,
  };
  render(<LiveTopBar {...props} />);
  return props;
}

describe("LiveTopBar", () => {
  it("shows the brand and model health summary", () => {
    renderBar();
    expect(screen.getByText("Sentinel")).toBeInTheDocument();
    expect(screen.getByText("healthy · 28 features")).toBeInTheDocument();
  });

  it("reflects the off state and invokes the toggle callback", () => {
    const props = renderBar();
    const toggle = screen.getByRole("button", { name: "Turn live refresh on" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    expect(toggle).toHaveTextContent("Off");

    fireEvent.click(toggle);
    expect(props.onToggleLiveRefresh).toHaveBeenCalledTimes(1);
  });

  it("reflects the on state", () => {
    renderBar({ liveRefreshEnabled: true });
    const toggle = screen.getByRole("button", { name: "Turn live refresh off" });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(toggle).toHaveTextContent("On");
  });

  it("renders last-updated and refresh errors when provided", () => {
    renderBar({
      liveRefreshEnabled: true,
      lastUpdatedText: "1:02:03 PM",
      refreshError: "Refresh failed: Stats unavailable",
    });
    expect(screen.getByText("Last updated 1:02:03 PM")).toBeInTheDocument();
    expect(screen.getByText("Refresh failed: Stats unavailable")).toBeInTheDocument();
  });

  it("hides the refresh error while live refresh is off", () => {
    renderBar({ liveRefreshEnabled: false, refreshError: "Refresh failed: nope" });
    expect(screen.queryByText("Refresh failed: nope")).not.toBeInTheDocument();
  });

  it("invokes onSimulate when the Simulate button is clicked", () => {
    const props = renderBar();
    fireEvent.click(screen.getByRole("button", { name: "Simulate" }));
    expect(props.onSimulate).toHaveBeenCalledTimes(1);
  });
});
