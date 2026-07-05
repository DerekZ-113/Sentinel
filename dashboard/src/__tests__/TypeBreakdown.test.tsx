/**
 * Tests for TypeBreakdown component.
 *
 * recharts renders nothing at zero size in jsdom, so assertions target the
 * panel chrome and the data mapping (labels), not SVG geometry.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import TypeBreakdown from "../components/TypeBreakdown";
import type { TypeStats } from "../services/api";

const byType: TypeStats[] = [
  {
    notification_type: "verification_request",
    total: 60,
    flagged: 10,
    suppressed: 50,
    fp_rate: 0.1,
    accuracy: 0.9,
  },
  {
    notification_type: "custom_future_type",
    total: 5,
    flagged: 5,
    suppressed: 0,
    fp_rate: null,
    accuracy: null,
  },
];

describe("TypeBreakdown", () => {
  it("renders the panel title", () => {
    render(<TypeBreakdown byType={byType} />);
    expect(screen.getByText("Alerts by Notification Type")).toBeInTheDocument();
  });

  it("renders with an empty type list", () => {
    render(<TypeBreakdown byType={[]} />);
    expect(screen.getByText("Alerts by Notification Type")).toBeInTheDocument();
  });

  it("applies the compact height override", () => {
    const { container } = render(<TypeBreakdown byType={byType} heightPx={260} />);
    const panel = container.firstElementChild as HTMLElement;
    expect(panel.style.height).toBe("260px");
  });

  it("defaults to the full-size panel height", () => {
    const { container } = render(<TypeBreakdown byType={byType} />);
    const panel = container.firstElementChild as HTMLElement;
    expect(panel.style.height).toBe("420px");
  });
});
