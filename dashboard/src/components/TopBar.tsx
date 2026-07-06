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
    <header className="sticky top-0 z-20 px-4 md:px-6 py-2.5 bg-gray-950/90 backdrop-blur-sm border-b border-gray-800">
      <div className="flex items-center gap-x-4 gap-y-1 flex-wrap text-xs">
        <div className="flex items-center gap-2.5">
          <div className="h-7 w-7 rounded-lg bg-blue-600 flex items-center justify-center text-xs font-bold">
            S
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight leading-tight">
              Sentinel
            </h1>
            <p className="text-gray-500 text-[10px] leading-tight">
              Fleet Alert Monitoring
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              health.status === "healthy" ? "bg-emerald-400" : "bg-red-400"
            }`}
          />
          <span className="text-gray-500 text-[11px]">
            {health.status} · {health.model_features} features
          </span>
        </div>

        {children}
      </div>
    </header>
  );
}
