/**
 * Row cells shared by AlertFeed (live) and LiveAlertFeed (demo).
 */

export function TypeBadge({
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

export function ConfidenceBar({ value }: { value: number }) {
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
