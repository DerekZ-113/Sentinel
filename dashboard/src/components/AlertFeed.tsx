import { useEffect, useRef, useState } from "react";
import { fetchAlerts } from "../services/api";
import type { AlertRecord } from "../services/api";
import { ConfidenceBar, TypeBadge } from "./alertRowParts";

const PAGE_SIZE = 20;

export default function AlertFeed() {
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const tableRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    fetchAlerts(PAGE_SIZE, page * PAGE_SIZE)
      .then((data) => {
        if (cancelled) return;
        setAlerts(data.alerts);
        setTotal(data.total);
        setLoading(false);
        tableRef.current?.scrollTo(0, 0);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [page]);

  function retry() {
    setLoading(true);
    setError(null);
    fetchAlerts(PAGE_SIZE, page * PAGE_SIZE)
      .then((data) => {
        setAlerts(data.alerts);
        setTotal(data.total);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-800/50 rounded-xl p-6 h-80 flex flex-col items-center justify-center gap-3">
        <p className="text-red-400 text-sm">{error}</p>
        <button
          onClick={retry}
          className="text-xs text-red-300 hover:text-white border border-red-700 px-3 py-1 rounded-lg transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-6 h-80 flex items-center justify-center text-gray-500">
        Loading alerts...
      </div>
    );
  }

  return (
    <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-5 overflow-hidden max-h-[420px] flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Recent Alerts</h2>
        <span className="text-xs text-gray-500">
          {total.toLocaleString()} total
        </span>
      </div>

      <div ref={tableRef} className="overflow-auto flex-1 overflow-x-auto">
        <table className="w-full text-sm min-w-[500px]">
          <thead>
            <tr className="text-gray-400 text-xs uppercase border-b border-gray-700/50">
              <th className="text-left py-2 pr-3">Vehicle</th>
              <th className="text-left py-2 pr-3">Type</th>
              <th className="text-left py-2 pr-3">Prediction</th>
              <th className="text-right py-2 pr-3">Confidence</th>
              <th className="text-left py-2 pr-3">Actual</th>
              <th className="text-center py-2">Correct</th>
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
                  className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
                >
                  <td className="py-2.5 pr-3 text-gray-300 font-mono text-xs">
                    {alert.vehicle_id}
                  </td>
                  <td className="py-2.5 pr-3">
                    <TypeBadge
                      type={alert.notification_type}
                      subtype={alert.notification_subtype}
                    />
                  </td>
                  <td className="py-2.5 pr-3">
                    {alert.needs_intervention_predicted ? (
                      <span className="text-red-400 font-medium">
                        ⚠ Flag
                      </span>
                    ) : (
                      <span className="text-emerald-400 font-medium">
                        ✓ Suppress
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 pr-3 text-right font-mono text-xs">
                    <ConfidenceBar value={alert.confidence} />
                  </td>
                  <td className="py-2.5 pr-3 text-xs">
                    {alert.needs_intervention_actual === null ? (
                      <span className="text-gray-600">—</span>
                    ) : alert.needs_intervention_actual ? (
                      <span className="text-red-300">Real</span>
                    ) : (
                      <span className="text-gray-400">FP</span>
                    )}
                  </td>
                  <td className="py-2.5 text-center">
                    {correct === null ? (
                      <span className="text-gray-600">—</span>
                    ) : correct ? (
                      <span className="text-emerald-400">✓</span>
                    ) : (
                      <span className="text-red-400">✗</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-700/50">
        <button
          onClick={() => setPage((p) => p - 1)}
          disabled={page === 0}
          className="text-xs text-gray-400 hover:text-white disabled:text-gray-600 disabled:cursor-not-allowed border border-gray-700 px-3 py-1 rounded-lg transition-colors"
        >
          Previous
        </button>
        <span className="text-xs text-gray-500">
          Page {page + 1} of {totalPages}
        </span>
        <button
          onClick={() => setPage((p) => p + 1)}
          disabled={page >= totalPages - 1}
          className="text-xs text-gray-400 hover:text-white disabled:text-gray-600 disabled:cursor-not-allowed border border-gray-700 px-3 py-1 rounded-lg transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  );
}

