/**
 * Row cells shared by AlertFeed (live) and LiveAlertFeed (demo).
 * Chip copy stays lowercase in textContent ("Flag"/"Suppress"/type names);
 * the uppercase presentation is CSS-only (DESIGN.md rule).
 */

export function TypeBadge({
  type,
  subtype,
}: {
  type: string;
  subtype: string | null;
}) {
  const colors: Record<string, string> = {
    verification_request: "text-tag-verif border-tag-verif/40 bg-tag-verif/14",
    stuck: "text-tag-stuck border-tag-stuck/40 bg-tag-stuck/14",
    emergency_vehicle_alert: "text-tag-ev border-tag-ev/40 bg-tag-ev/14",
    speed_anomaly: "text-tag-speed border-tag-speed/40 bg-tag-speed/14",
    impact_l0: "text-tag-impact border-tag-impact/40 bg-tag-impact/14",
    passenger_assist: "text-tag-pax border-tag-pax/40 bg-tag-pax/14",
  };

  return (
    <span
      className="inline-block max-w-full"
      title={subtype ? `${type}/${subtype}` : type}
    >
      <span
        className={`inline-block px-1.5 py-0.5 rounded-xs text-[9.5px] uppercase tracking-[0.05em] border whitespace-nowrap ${
          colors[type] || "text-ink-mid border-hairline-2 bg-inset"
        }`}
      >
        {type.replace(/_/g, " ")}
      </span>
      {subtype && (
        <span className="block text-[9.5px] text-ink-low truncate">
          {subtype.replace(/_/g, " ")}
        </span>
      )}
    </span>
  );
}

export function VerdictChip({ flagged }: { flagged: boolean }) {
  return (
    <span
      className={`inline-block px-1.5 py-0.5 rounded-xs border text-[9.5px] uppercase tracking-[0.05em] font-medium whitespace-nowrap ${
        flagged
          ? "text-crit border-crit/40 bg-crit/14"
          : "text-ok border-ok/40 bg-ok/14"
      }`}
    >
      {flagged ? "Flag" : "Suppress"}
    </span>
  );
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const width = `${pct}%`;
  const color =
    pct >= 90 ? "bg-ok" : pct >= 70 ? "bg-warn" : "bg-crit";

  return (
    <span className="inline-flex w-16 flex-col items-end gap-[3px] align-middle">
      <span className="text-[10px] leading-none tabular-nums text-ink-data">
        {pct}%
      </span>
      <span className="block h-[3px] w-full overflow-hidden bg-inset">
        <span className={`block h-full ${color}`} style={{ width }} />
      </span>
    </span>
  );
}
