/**
 * Tests for the row cells shared by AlertFeed and LiveAlertFeed.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConfidenceBar, TypeBadge } from "../components/alertRowParts";

describe("TypeBadge", () => {
  it("renders the type with the subtype demoted to a second line", () => {
    render(<TypeBadge type="verification_request" subtype="object_query" />);

    expect(screen.getByText("verification request")).toBeInTheDocument();
    expect(screen.getByText("object query")).toBeInTheDocument();
    // Full taxonomy value survives as a tooltip on the wrapper
    expect(
      screen.getByTitle("verification_request/object_query")
    ).toBeInTheDocument();
  });

  it("renders the badge alone when there is no subtype", () => {
    render(<TypeBadge type="stuck" subtype={null} />);

    expect(screen.getByText("stuck")).toBeInTheDocument();
    expect(screen.getByTitle("stuck")).toBeInTheDocument();
    expect(screen.queryByText("stuck/")).not.toBeInTheDocument();
  });
});

describe("ConfidenceBar", () => {
  it("shows the percentage inside the bar", () => {
    render(<ConfidenceBar value={0.92} />);
    expect(screen.getByText("92%")).toBeInTheDocument();
  });

  it("rounds to whole percentages", () => {
    render(<ConfidenceBar value={0.876} />);
    expect(screen.getByText("88%")).toBeInTheDocument();
  });
});
