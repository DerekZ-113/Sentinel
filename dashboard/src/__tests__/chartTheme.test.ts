/**
 * Guard: chart hexes must mirror the @theme tokens in src/index.css.
 * (Two sources of truth by necessity — SVG attrs can't resolve var().)
 */

import { describe, it, expect } from "vitest";
// @ts-expect-error -- node types are not installed in the app tsconfig;
// vitest executes on node, where this resolves fine. (`?raw` can't be used:
// vitest's CSS pipeline intercepts .css imports and returns an empty string.)
import { readFileSync } from "node:fs";
import { CHART } from "../components/chartTheme";

describe("chartTheme", () => {
  it("mirrors the @theme tokens in index.css", () => {
    // vitest runs with cwd = dashboard/ (npm test / npx vitest from dashboard)
    const css = readFileSync("src/index.css", "utf8").toLowerCase();
    for (const [name, hex] of Object.entries(CHART)) {
      expect(css, `token ${name} (${hex}) missing from index.css`).toContain(
        hex.toLowerCase()
      );
    }
  });
});
