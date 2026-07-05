import type { ReactNode } from "react";
import type { StatsResponse } from "../services/api";
import AnimatedNumber from "./AnimatedNumber";

interface OverviewCardsProps {
  stats: StatsResponse;
  /** Overrides the "Last Nh" subtitle (demo mode shows "Current shift"). */
  windowLabel?: string;
}

function StatCard({
  label,
  value,
  subtitle,
  color = "text-white",
}: {
  label: string;
  value: ReactNode;
  subtitle?: string;
  color?: string;
}) {
  return (
    <div className="bg-gray-800/60 border border-gray-700/50 rounded-xl p-4">
      <p className="text-gray-400 text-sm font-medium">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color}`}>{value}</p>
      {subtitle && <p className="text-gray-500 text-xs mt-1">{subtitle}</p>}
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
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <StatCard
        label="Total Alerts"
        value={<AnimatedNumber value={stats.total_alerts} />}
        subtitle={windowLabel ?? `Last ${stats.time_window_hours}h`}
      />
      <StatCard
        label="Flagged for Review"
        value={<AnimatedNumber value={stats.total_flagged} />}
        subtitle="Predicted as needing intervention"
        color="text-red-400"
      />
      <StatCard
        label="Suppressed"
        value={<AnimatedNumber value={stats.total_suppressed} />}
        subtitle={`${suppressionRate}% of alerts filtered`}
        color="text-emerald-400"
      />
      <StatCard
        label="Model FP Rate"
        value={
          fpRate !== null ? (
            <AnimatedNumber
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
            ? "text-emerald-400"
            : fpRate !== null && fpRate < 0.5
            ? "text-yellow-400"
            : "text-red-400"
        }
      />
    </div>
  );
}
