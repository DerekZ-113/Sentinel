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
    <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-5">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">FP Rate Over Time</h2>
        <span className="text-xs text-gray-500">
          {windowLabel ?? `Last ${data.time_window_hours} hours`}
        </span>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mb-3">
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block"
            style={{
              width: 10,
              height: 10,
              background: "#ef4444",
              borderRadius: 2,
            }}
          />
          <span className="text-xs text-gray-400">FP Rate</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block"
            style={{
              width: 10,
              height: 10,
              background: "#10b981",
              borderRadius: 2,
            }}
          />
          <span className="text-xs text-gray-400">Accuracy</span>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={chartHeight}>
        <ComposedChart data={chartData}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(55,65,81,0.3)"
          />
          <XAxis
            dataKey="time_label"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${v}%`}
            domain={[0, 100]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              color: "#f3f4f6",
              fontSize: "13px",
            }}
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
            stroke="#ef4444"
            strokeWidth={2}
            fill="#ef4444"
            fillOpacity={0.08}
            dot={{ r: 3, fill: "#ef4444" }}
            connectNulls
            isAnimationActive={animate}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#10b981"
            strokeWidth={2}
            dot={{ r: 3, fill: "#10b981" }}
            connectNulls
            isAnimationActive={animate}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
