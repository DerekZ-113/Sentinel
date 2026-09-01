/**
 * Mode-neutral top bar shell: brand block + model-health summary, with
 * mode-specific controls injected as children into the same flex row.
 * Imports nothing from the demo layer so it is safe in live bundles.
 */

import type { ReactNode } from "react";
import type { HealthResponse } from "../services/api";

interface TopBarProps {
  health: Pick<HealthResponse, "status" | "model_features">;
  children?: ReactNode;
}

export default function TopBar({ health, children }: TopBarProps) {
  return (
    <header className="sticky top-0 z-20 px-4 md:px-6 py-2.5 bg-header border-b border-hairline">
      <div className="flex items-center gap-x-4 gap-y-1 flex-wrap text-xs">
        <div className="leading-tight">
          <h1 className="text-xs font-semibold uppercase tracking-[0.18em] text-ink">
            Sentinel
          </h1>
          <p className="text-ink-micro text-[9px] uppercase tracking-[0.1em]">
            Fleet Alert Monitoring
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 ${
              health.status === "healthy" ? "bg-ok" : "bg-crit"
            }`}
          />
          <span className="text-ink-low text-[11px]">
            {health.status} · {health.model_features} features
          </span>
        </div>

        {children}
      </div>
    </header>
  );
}
