/**
 * Right-rail summary of model health: status + three key stats.
 * The full histogram/donut view opens in a drawer via onExpand.
 * (Suppression rate already lives in the KPI strip.)
 */

import type { ModelHealthResponse } from "../services/api";
import { STATUS_CONFIG } from "./modelHealthStatus";
import { MiniCard } from "./ModelHealthView";

interface CompactModelHealthCardProps {
  data: ModelHealthResponse;
  onExpand?: () => void;
}

export default function CompactModelHealthCard({
  data,
  onExpand,
}: CompactModelHealthCardProps) {
  const statusCfg = STATUS_CONFIG[data.status] || STATUS_CONFIG.warning;

  return (
    <div className="bg-panel border border-hairline rounded-xs p-4 space-y-2.5">
      <div className="flex items-center justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-mid">
          Model Health
        </h2>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 ${statusCfg.bg}`} />
          <span className={`text-[11px] font-medium uppercase tracking-[0.1em] ${statusCfg.color}`}>
            {statusCfg.label}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <MiniCard
          label="Predictions"
          value={data.total_predictions.toLocaleString()}
        />
        <MiniCard
          label="Avg Confidence"
          value={
            data.avg_confidence !== null
              ? `${(data.avg_confidence * 100).toFixed(1)}%`
              : "N/A"
          }
          color={
            data.avg_confidence !== null && data.avg_confidence >= 0.9
              ? "text-ok"
              : data.avg_confidence !== null && data.avg_confidence >= 0.7
              ? "text-warn"
              : "text-crit"
          }
        />
        <MiniCard
          label="Accuracy"
          value={
            data.accuracy !== null
              ? `${(data.accuracy * 100).toFixed(1)}%`
              : "N/A"
          }
          color={
            data.accuracy !== null && data.accuracy >= 0.8
              ? "text-ok"
              : data.accuracy !== null && data.accuracy >= 0.6
              ? "text-warn"
              : "text-crit"
          }
        />
      </div>

      <button
        onClick={onExpand}
        className="w-full text-[10px] uppercase tracking-[0.1em] text-ink-mid hover:text-ink border border-hairline-2 rounded-xs py-1.5 transition-colors"
      >
        Details
      </button>
    </div>
  );
}
