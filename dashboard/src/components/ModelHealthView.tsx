import type { ModelHealthResponse } from "../services/api";
import { STATUS_CONFIG } from "./modelHealthStatus";
import { CHART, CONF_COLORS, AXIS_TICK, TOOLTIP_STYLE } from "./chartTheme";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

interface ModelHealthViewProps {
  data: ModelHealthResponse;
  /** Disable recharts animations for frequently-updating demo data. */
  animate?: boolean;
}

export default function ModelHealthView({ data, animate = true }: ModelHealthViewProps) {
  const statusCfg = STATUS_CONFIG[data.status] || STATUS_CONFIG.warning;

  const confData = [
    { name: "High (≥90%)", value: data.confidence_buckets.high, fill: CONF_COLORS.high },
    { name: "Medium (70-90%)", value: data.confidence_buckets.medium, fill: CONF_COLORS.medium },
    { name: "Low (<70%)", value: data.confidence_buckets.low, fill: CONF_COLORS.low },
  ];

  const predSplit = [
    { name: "Suppressed", value: Math.round(data.pct_suppressed * 10) / 10, fill: CHART.ok },
    { name: "Flagged", value: Math.round(data.pct_flagged * 10) / 10, fill: CHART.crit },
  ];

  return (
    <div className="bg-panel border border-hairline rounded-xs p-5 space-y-5">
      {/* Header + Status */}
      <div className="flex items-center justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-mid">
          Model Health
        </h2>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 ${statusCfg.bg}`} />
          <span className={`text-[11px] font-medium uppercase tracking-[0.1em] ${statusCfg.color}`}>
            {statusCfg.label}
          </span>
        </div>
      </div>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MiniCard
          label="Predictions"
          value={data.total_predictions.toLocaleString()}
        />
        <MiniCard
          label="Avg Confidence"
          value={data.avg_confidence !== null ? `${(data.avg_confidence * 100).toFixed(1)}%` : "N/A"}
          color={
            data.avg_confidence !== null && data.avg_confidence >= 0.9
              ? "text-ok"
              : data.avg_confidence !== null && data.avg_confidence >= 0.7
              ? "text-warn"
              : "text-crit"
          }
        />
        <MiniCard
          label="Accuracy"
          value={data.accuracy !== null ? `${(data.accuracy * 100).toFixed(1)}%` : "N/A"}
          color={
            data.accuracy !== null && data.accuracy >= 0.8
              ? "text-ok"
              : data.accuracy !== null && data.accuracy >= 0.6
              ? "text-warn"
              : "text-crit"
          }
        />
        <MiniCard
          label="Suppression Rate"
          value={`${data.pct_suppressed.toFixed(1)}%`}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Confidence Distribution */}
        <div>
          <p className="text-[10px] uppercase tracking-[0.1em] text-ink-micro mb-2">Confidence Distribution</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={confData} barSize={40}>
              <XAxis
                dataKey="name"
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="value" radius={[2, 2, 0, 0]} isAnimationActive={animate}>
                {confData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Prediction Split */}
        <div>
          <p className="text-[10px] uppercase tracking-[0.1em] text-ink-micro mb-2">Prediction Split</p>
          <div className="flex items-center justify-center gap-6">
            <ResponsiveContainer width={160} height={160}>
              <PieChart>
                <Pie
                  data={predSplit}
                  cx="50%"
                  cy="50%"
                  innerRadius={45}
                  outerRadius={70}
                  dataKey="value"
                  stroke="none"
                  isAnimationActive={animate}
                >
                  {predSplit.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2">
              {predSplit.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                  <span
                    className="h-2 w-2"
                    style={{ backgroundColor: item.fill }}
                  />
                  <span className="text-[11px] text-ink-mid">
                    {item.name}: <span className="text-ink font-medium tabular-nums">{item.value}%</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function MiniCard({
  label,
  value,
  color = "text-ink",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-inset border border-line rounded-xs px-3 py-2.5">
      <p className="text-ink-micro text-[10px] uppercase tracking-[0.1em]">{label}</p>
      <p className={`text-base font-medium tabular-nums ${color}`}>{value}</p>
    </div>
  );
}
