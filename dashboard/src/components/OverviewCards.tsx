import type { ReactNode } from "react";
import type { StatsResponse } from "../services/api";
import ValueFlash from "./ValueFlash";

interface OverviewCardsProps {
  stats: StatsResponse;
  /** Overrides the "Last Nh" subtitle (demo mode shows "Current shift"). */
  windowLabel?: string;
}

function StatCard({
  label,
  value,
  subtitle,
  color = "text-ink",
}: {
  label: string;
  value: ReactNode;
  subtitle?: string;
  color?: string;
}) {
  return (
    <div className="bg-panel border border-hairline rounded-xs px-3.5 py-2.5">
      <p className="text-ink-micro text-[10px] uppercase tracking-[0.1em]">{label}</p>
      <p className={`text-[22px] font-medium tabular-nums mt-0.5 ${color}`}>{value}</p>
      {subtitle && (
        <p className="text-ink-low text-[10px] mt-0.5 truncate">{subtitle}</p>
      )}
    </div>
  );
}

export default function OverviewCards({ stats, windowLabel }: OverviewCardsProps) {
  const fpRate = stats.overall_fp_rate;

  const suppressionRate =
    stats.total_alerts > 0
      ? ((stats.total_suppressed / stats.total_alerts) * 100).toFixed(1)
      : "0";

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <StatCard
        label="Total Alerts"
        value={<ValueFlash value={stats.total_alerts} />}
        subtitle={windowLabel ?? `Last ${stats.time_window_hours}h`}
      />
      <StatCard
        label="Flagged for Review"
        value={<ValueFlash value={stats.total_flagged} />}
        subtitle="Predicted as needing intervention"
        color="text-crit"
      />
      <StatCard
        label="Suppressed"
        value={<ValueFlash value={stats.total_suppressed} />}
        subtitle={`${suppressionRate}% of alerts filtered`}
        color="text-ok"
      />
      <StatCard
        label="Model FP Rate"
        value={
          fpRate !== null ? (
            <ValueFlash
              value={fpRate * 100}
              format={(n) => `${n.toFixed(1)}%`}
            />
          ) : (
            "N/A"
          )
        }
        subtitle="Among flagged alerts"
        color={
          fpRate !== null && fpRate < 0.3
            ? "text-ok"
            : fpRate !== null && fpRate < 0.5
            ? "text-warn"
            : "text-crit"
        }
      />
    </div>
  );
}
