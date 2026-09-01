import type { FPOverTimeResponse } from "../services/api";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { CHART, AXIS_TICK, TOOLTIP_STYLE } from "./chartTheme";

function formatHour(isoString: string): string {
  const d = new Date(isoString);
  const h = d.getHours();
  if (h === 0) return "12 AM";
  if (h === 12) return "12 PM";
  return h > 12 ? `${h - 12} PM` : `${h} AM`;
}

interface FPRateChartViewProps {
  data: FPOverTimeResponse;
  /** Header label; defaults to "Last N hours" from the payload. */
  windowLabel?: string;
  /** X-axis tick label; defaults to hour-of-day. */
  tickFormatter?: (iso: string) => string;
  /** Disable recharts animations for frequently-updating demo data. */
  animate?: boolean;
  chartHeight?: number;
}

export default function FPRateChartView({
  data,
  windowLabel,
  tickFormatter = formatHour,
  animate = true,
  chartHeight = 240,
}: FPRateChartViewProps) {
  const chartData = data.buckets.map((b) => ({
    time_label: tickFormatter(b.time),
    fp_rate: b.fp_rate !== null ? Math.round(b.fp_rate * 1000) / 10 : null,
    accuracy: b.accuracy !== null ? Math.round(b.accuracy * 1000) / 10 : null,
    total: b.total,
    flagged: b.flagged,
    suppressed: b.suppressed,
  }));

  return (
    <div className="bg-panel border border-hairline rounded-xs p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-2.5">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-mid">
          FP Rate Over Time
        </h2>
        <span className="text-[10px] text-ink-low">
          {windowLabel ?? `Last ${data.time_window_hours} hours`}
        </span>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mb-2">
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block"
            style={{
              width: 8,
              height: 8,
              background: CHART.crit,
              borderRadius: 1,
            }}
          />
          <span className="text-[10px] text-ink-mid">FP Rate</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block"
            style={{
              width: 8,
              height: 8,
              background: CHART.ok,
              borderRadius: 1,
            }}
          />
          <span className="text-[10px] text-ink-mid">Accuracy</span>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={chartHeight}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART.line} />
          <XAxis
            dataKey="time_label"
            tick={AXIS_TICK}
            axisLine={{ stroke: CHART.hairlineStrong }}
            tickLine={false}
          />
          <YAxis
            tick={AXIS_TICK}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${v}%`}
            domain={[0, 100]}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            cursor={{ stroke: "rgba(255,255,255,0.1)" }}
            formatter={(value, name) => [
              `${Number(value ?? 0).toFixed(1)}%`,
              name === "fp_rate" ? "FP Rate" : "Accuracy",
            ]}
            labelFormatter={(label) => String(label)}
          />
          <Area
            type="monotone"
            dataKey="fp_rate"
            stroke={CHART.crit}
            strokeWidth={2}
            fill={CHART.crit}
            fillOpacity={0.08}
            dot={{ r: 3, fill: CHART.crit }}
            connectNulls
            isAnimationActive={animate}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke={CHART.ok}
            strokeWidth={2}
            dot={{ r: 3, fill: CHART.ok }}
            connectNulls
            isAnimationActive={animate}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
