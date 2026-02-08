import { useEffect, useState } from "react";
import { fetchAlerts } from "../services/api";
import type { AlertRecord } from "../services/api";

export default function AlertFeed() {
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAlerts(20).then((data) => {
      setAlerts(data.alerts);
      setTotal(data.total);
      setLoading(false);
    });
  }, []);

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

      <div className="overflow-auto flex-1">
        <table className="w-full text-sm">
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
    </div>
  );
}

function TypeBadge({
  type,
  subtype,
}: {
  type: string;
  subtype: string | null;
}) {
  const colors: Record<string, string> = {
    verification_request: "bg-blue-900/50 text-blue-300 border-blue-800/50",
    stuck: "bg-amber-900/50 text-amber-300 border-amber-800/50",
    emergency_vehicle_alert:
      "bg-red-900/50 text-red-300 border-red-800/50",
    speed_anomaly:
      "bg-purple-900/50 text-purple-300 border-purple-800/50",
    impact_l0: "bg-orange-900/50 text-orange-300 border-orange-800/50",
    passenger_assist: "bg-cyan-900/50 text-cyan-300 border-cyan-800/50",
  };

  const label = subtype ? `${type}/${subtype}` : type;
  const display = label.replace(/_/g, " ");

  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs border ${
        colors[type] || "bg-gray-800 text-gray-300 border-gray-700"
      }`}
    >
      {display}
    </span>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const width = `${pct}%`;
  const color =
    pct >= 90
      ? "bg-emerald-500"
      : pct >= 70
      ? "bg-yellow-500"
      : "bg-red-500";

  return (
    <div className="flex items-center gap-2 justify-end">
      <span className="text-gray-400">{pct}%</span>
      <div className="w-16 h-1.5 bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width }} />
      </div>
    </div>
  );
}
