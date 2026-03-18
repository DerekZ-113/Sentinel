/**
 * Tests for DemoBanner component.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DemoBanner from "../components/DemoBanner";

describe("DemoBanner", () => {
  const originalEnv = import.meta.env.VITE_DEMO_MODE;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    import.meta.env.VITE_DEMO_MODE = originalEnv;
  });

  it("renders in demo mode", () => {
    import.meta.env.VITE_DEMO_MODE = "true";
    render(<DemoBanner />);
    expect(screen.getByText(/Demo mode/)).toBeInTheDocument();
  });

  it("shows GitHub link", () => {
    import.meta.env.VITE_DEMO_MODE = "true";
    render(<DemoBanner />);
    const link = screen.getByText("Run with Docker");
    expect(link).toHaveAttribute("href", "https://github.com/DerekZ-113/Sentinel");
  });

  it("dismisses on x click", async () => {
    import.meta.env.VITE_DEMO_MODE = "true";
    const user = userEvent.setup();
    render(<DemoBanner />);
    expect(screen.getByText(/Demo mode/)).toBeInTheDocument();
    await user.click(screen.getByLabelText("Dismiss"));
    expect(screen.queryByText(/Demo mode/)).not.toBeInTheDocument();
  });

  it("does not render when not in demo mode", () => {
    import.meta.env.VITE_DEMO_MODE = "false";
    render(<DemoBanner />);
    expect(screen.queryByText(/Demo mode/)).not.toBeInTheDocument();
  });
});
