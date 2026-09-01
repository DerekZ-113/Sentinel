import { useEffect, useRef, useState } from "react";
import { fetchAlerts } from "../services/api";
import type { AlertRecord } from "../services/api";
import { ConfidenceBar, TypeBadge, VerdictChip } from "./alertRowParts";
import { IconCheck, IconCross } from "./icons";

const PAGE_SIZE = 50;

interface AlertFeedProps {
  refreshToken?: number;
  /** Constant for a mount — App remounts the feed via key when it changes. */
  filterType?: string | null;
}

export default function AlertFeed({
  refreshToken = 0,
  filterType = null,
}: AlertFeedProps) {
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const tableRef = useRef<HTMLDivElement>(null);
  const hasLoadedRef = useRef(false);
  const lastPageRef = useRef(page);

  useEffect(() => {
    let cancelled = false;
    const isRefreshOnly = hasLoadedRef.current && lastPageRef.current === page;

    fetchAlerts(PAGE_SIZE, page * PAGE_SIZE, filterType ?? undefined)
      .then((data) => {
        if (cancelled) return;
        setAlerts(data.alerts);
        setTotal(data.total);
        setError(null);
        setRefreshError(null);
        setLoading(false);
        hasLoadedRef.current = true;
        lastPageRef.current = page;
        if (!isRefreshOnly) {
          tableRef.current?.scrollTo(0, 0);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Unknown error";
        if (isRefreshOnly) {
          setRefreshError(`Refresh failed: ${message}`);
        } else {
          setError(message);
        }
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [page, refreshToken, filterType]);

  function goToPage(nextPage: number) {
    setLoading(true);
    setError(null);
    setRefreshError(null);
    setPage(nextPage);
  }

  function retry() {
    setLoading(true);
    setError(null);
    setRefreshError(null);
    fetchAlerts(PAGE_SIZE, page * PAGE_SIZE, filterType ?? undefined)
      .then((data) => {
        setAlerts(data.alerts);
        setTotal(data.total);
        setLoading(false);
        hasLoadedRef.current = true;
        lastPageRef.current = page;
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
        setLoading(false);
      });
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  if (error) {
    return (
      <div className="bg-crit/10 border border-crit/40 rounded-xs p-6 h-80 flex flex-col items-center justify-center gap-3">
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

  if (loading) {
    return (
      <div className="bg-panel border border-hairline rounded-xs p-6 h-80 flex items-center justify-center text-ink-low">
        Loading alerts...
      </div>
    );
  }

  return (
    <div className="bg-panel border border-hairline rounded-xs p-5 overflow-hidden max-h-[420px] lg:max-h-none lg:h-full min-h-0 flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-mid">
          Recent Alerts
        </h2>
        <span className="text-[10px] text-ink-low tabular-nums">
          {total.toLocaleString()} total
        </span>
      </div>

      {refreshError && (
        <p className="mb-3 rounded-xs border border-warn/40 bg-warn/10 px-3 py-1.5 text-xs text-warn">
          {refreshError}
        </p>
      )}

      <div ref={tableRef} className="overflow-auto flex-1">
        <table className="w-full text-sm table-fixed">
          <thead>
            <tr className="text-ink-micro text-[10px] uppercase tracking-[0.1em]">
              <th className="sticky top-0 z-10 bg-panel border-b border-hairline-2 text-left py-2 pr-3 w-28">
                Vehicle
              </th>
              <th className="sticky top-0 z-10 bg-panel border-b border-hairline-2 text-left py-2 pr-3">
                Type
              </th>
              <th className="sticky top-0 z-10 bg-panel border-b border-hairline-2 text-left py-2 pr-3 w-24">
                Prediction
              </th>
              <th className="sticky top-0 z-10 bg-panel border-b border-hairline-2 text-right py-2 pr-3 w-20">
                Confidence
              </th>
              <th className="sticky top-0 z-10 bg-panel border-b border-hairline-2 text-left py-2 pr-3 w-14">
                Actual
              </th>
              <th className="sticky top-0 z-10 bg-panel border-b border-hairline-2 text-center py-2 w-16">
                Correct
              </th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((alert) => {
              const correct =
                alert.needs_intervention_actual !== null
                  ? alert.needs_intervention_predicted ===
                    alert.needs_intervention_actual
                  : null;

              return (
                <tr
                  key={alert.id}
                  className="border-b border-line hover:bg-ink/5 transition-colors"
                >
                  <td className="py-2.5 pr-3 text-ink-data text-xs">
                    {alert.vehicle_id}
                  </td>
                  <td className="py-2.5 pr-3">
                    <TypeBadge
                      type={alert.notification_type}
                      subtype={alert.notification_subtype}
                    />
                  </td>
                  <td className="py-2.5 pr-3">
                    <VerdictChip flagged={alert.needs_intervention_predicted} />
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono text-xs">
                    <ConfidenceBar value={alert.confidence} />
                  </td>
                  <td className="py-2.5 pr-3 text-xs">
                    {alert.needs_intervention_actual === null ? (
                      <span className="text-ink-low">—</span>
                    ) : alert.needs_intervention_actual ? (
                      <span className="text-crit">Real</span>
                    ) : (
                      <span className="text-ink-mid">FP</span>
                    )}
                  </td>
                  <td className="py-2.5 text-center">
                    {correct === null ? (
                      <span className="text-ink-low">—</span>
                    ) : correct ? (
                      <span
                        role="img"
                        aria-label="correct"
                        className="inline-flex justify-center text-ok"
                      >
                        <IconCheck size={11} />
                      </span>
                    ) : (
                      <span
                        role="img"
                        aria-label="incorrect"
                        className="inline-flex justify-center text-crit"
                      >
                        <IconCross size={11} />
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between pt-3 border-t border-hairline">
        <button
          onClick={() => goToPage(page - 1)}
          disabled={page === 0}
          className="text-[10px] uppercase tracking-[0.1em] text-ink-mid hover:text-ink disabled:text-ink-low disabled:cursor-not-allowed border border-hairline-2 px-3 py-1 rounded-xs transition-colors"
        >
          Previous
        </button>
        <span className="text-[10px] text-ink-low tabular-nums">
          Page {page + 1} of {totalPages}
        </span>
        <button
          onClick={() => goToPage(page + 1)}
          disabled={page >= totalPages - 1}
          className="text-[10px] uppercase tracking-[0.1em] text-ink-mid hover:text-ink disabled:text-ink-low disabled:cursor-not-allowed border border-hairline-2 px-3 py-1 rounded-xs transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  );
}

