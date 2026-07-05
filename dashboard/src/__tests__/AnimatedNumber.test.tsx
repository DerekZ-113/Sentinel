/**
 * Tests for AnimatedNumber.
 */

import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import AnimatedNumber from "../components/AnimatedNumber";

describe("AnimatedNumber", () => {
  it("renders the final value immediately on mount", () => {
    render(<AnimatedNumber value={1234} />);
    expect(screen.getByText("1,234")).toBeInTheDocument();
  });

  it("tweens to the new value after an update", async () => {
    const { rerender } = render(<AnimatedNumber value={1000} />);
    rerender(<AnimatedNumber value={2000} />);

    await waitFor(
      () => {
        expect(screen.getByText("2,000")).toBeInTheDocument();
      },
      { timeout: 2000 }
    );
  });

  it("applies a custom format", () => {
    render(<AnimatedNumber value={25} format={(n) => `${n.toFixed(1)}%`} />);
    expect(screen.getByText("25.0%")).toBeInTheDocument();
  });
});
