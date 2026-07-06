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

  return (
    <span
      className="inline-block max-w-full"
      title={subtype ? `${type}/${subtype}` : type}
    >
      <span
        className={`inline-block px-2 py-0.5 rounded text-xs border whitespace-nowrap ${
          colors[type] || "bg-gray-800 text-gray-300 border-gray-700"
        }`}
      >
        {type.replace(/_/g, " ")}
      </span>
      {subtype && (
        <span className="block text-[10px] text-gray-500 truncate">
          {subtype.replace(/_/g, " ")}
        </span>
      )}
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
    <div className="relative inline-block align-middle w-16 h-4 bg-gray-700 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full ${color} opacity-70`}
        style={{ width }}
      />
      <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-white leading-none">
        {pct}%
      </span>
    </div>
  );
}
