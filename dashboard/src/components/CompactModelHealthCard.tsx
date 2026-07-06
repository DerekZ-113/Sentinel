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
    <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-4 space-y-2.5">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-white">Model Health</h2>
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${statusCfg.bg}`} />
          <span className={`text-sm font-medium ${statusCfg.color}`}>
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
              ? "text-emerald-400"
              : data.avg_confidence !== null && data.avg_confidence >= 0.7
              ? "text-yellow-400"
              : "text-red-400"
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
              ? "text-emerald-400"
              : data.accuracy !== null && data.accuracy >= 0.6
              ? "text-yellow-400"
              : "text-red-400"
          }
        />
      </div>

      <button
        onClick={onExpand}
        className="w-full text-xs text-gray-400 hover:text-white border border-gray-700 rounded-lg py-1.5 transition-colors"
      >
        Details
      </button>
    </div>
  );
}
