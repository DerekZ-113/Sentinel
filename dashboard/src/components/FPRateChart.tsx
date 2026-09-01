import { useEffect, useRef, useState } from "react";
import { fetchFPOverTime } from "../services/api";
import type { FPOverTimeResponse } from "../services/api";
import FPRateChartView from "./FPRateChartView";

interface FPRateChartProps {
  refreshToken?: number;
  chartHeight?: number;
}

export default function FPRateChart({
  refreshToken = 0,
  chartHeight,
}: FPRateChartProps) {
  const [data, setData] = useState<FPOverTimeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const isRefresh = hasLoadedRef.current;
    fetchFPOverTime()
      .then((nextData) => {
        if (cancelled) return;
        setData(nextData);
        setError(null);
        setRefreshError(null);
        hasLoadedRef.current = true;
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Unknown error";
        if (isRefresh) {
          setRefreshError(`Refresh failed: ${message}`);
        } else {
          setError(message);
        }
      });
    return () => { cancelled = true; };
  }, [refreshToken]);

  function retry() {
    setError(null);
    setRefreshError(null);
    setData(null);
    hasLoadedRef.current = false;
    fetchFPOverTime()
      .then((nextData) => {
        setData(nextData);
        setError(null);
        hasLoadedRef.current = true;
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
      });
  }

  if (error) {
    return (
      <div className="bg-crit/10 border border-crit/40 rounded-xs p-6 flex flex-col items-center justify-center gap-3 h-64">
        <p className="text-crit text-sm">{error}</p>
        <button
          onClick={retry}
          className="text-[10px] uppercase tracking-[0.1em] text-crit hover:bg-crit/10 border border-crit/50 px-3 py-1 rounded-xs transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-panel border border-hairline rounded-xs p-6 flex items-center justify-center text-ink-low h-64">
        Loading FP rate trend...
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {refreshError && (
        <p className="rounded-xs border border-warn/40 bg-warn/10 px-3 py-1.5 text-xs text-warn">
          {refreshError}
        </p>
      )}
      <FPRateChartView data={data} chartHeight={chartHeight} />
    </div>
  );
}
