/**
 * Tests for ValueFlash — values snap synchronously (no tween), with a
 * flash class armed only after a change.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ValueFlash from "../components/ValueFlash";

describe("ValueFlash", () => {
  it("renders the exact value immediately on mount, without the flash class", () => {
    render(<ValueFlash value={1234} />);
    const el = screen.getByText("1,234");
    expect(el).toBeInTheDocument();
    expect(el).not.toHaveClass("value-flash");
  });

  it("renders the new exact value synchronously after an update", () => {
    const { rerender } = render(<ValueFlash value={1000} />);
    expect(screen.getByText("1,000")).toBeInTheDocument();

    rerender(<ValueFlash value={2000} />);
    // No waitFor: the value snaps in the same render
    expect(screen.getByText("2,000")).toBeInTheDocument();
    expect(screen.queryByText("1,000")).not.toBeInTheDocument();
  });

  it("applies a custom format", () => {
    render(<ValueFlash value={25} format={(n) => `${n.toFixed(1)}%`} />);
    expect(screen.getByText("25.0%")).toBeInTheDocument();
  });

  it("arms the flash class only after a change", async () => {
    const { rerender } = render(<ValueFlash value={1000} />);
    expect(screen.getByText("1,000")).not.toHaveClass("value-flash");

    rerender(<ValueFlash value={2000} />);
    // The flash class lands after the change-detection effect commits
    expect(await screen.findByText("2,000")).toBeInTheDocument();
    expect(screen.getByText("2,000")).toHaveClass("value-flash");
  });
});
