import { useEffect, useState } from "react";
import { fetchModelHealth } from "../services/api";
import type { ModelHealthResponse } from "../services/api";
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

const STATUS_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  healthy: { color: "text-emerald-400", bg: "bg-emerald-400", label: "Healthy" },
  warning: { color: "text-yellow-400", bg: "bg-yellow-400", label: "Warning" },
  degraded: { color: "text-red-400", bg: "bg-red-400", label: "Degraded" },
};

const CONF_COLORS = {
  high: "#10b981",
  medium: "#f59e0b",
  low: "#ef4444",
};

export default function ModelHealth() {
  const [data, setData] = useState<ModelHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadHealth = () => {
    setError(null);
    setData(null);
    fetchModelHealth()
      .then(setData)
      .catch((err) => setError(err.message));
  };

  useEffect(() => {
    loadHealth();
  }, []);

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-800/50 rounded-xl p-6 flex flex-col items-center justify-center gap-3 h-64">
        <p className="text-red-400 text-sm">{error}</p>
        <button
          onClick={loadHealth}
          className="text-xs text-red-300 hover:text-white border border-red-700 px-3 py-1 rounded-lg transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-6 flex items-center justify-center text-gray-500 h-64">
        Loading model health...
      </div>
    );
  }

  const statusCfg = STATUS_CONFIG[data.status] || STATUS_CONFIG.warning;

  const confData = [
    { name: "High (≥90%)", value: data.confidence_buckets.high, fill: CONF_COLORS.high },
    { name: "Medium (70-90%)", value: data.confidence_buckets.medium, fill: CONF_COLORS.medium },
    { name: "Low (<70%)", value: data.confidence_buckets.low, fill: CONF_COLORS.low },
  ];

  const predSplit = [
    { name: "Suppressed", value: Math.round(data.pct_suppressed * 10) / 10, fill: "#10b981" },
    { name: "Flagged", value: Math.round(data.pct_flagged * 10) / 10, fill: "#ef4444" },
  ];

  return (
    <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-5 space-y-5">
      {/* Header + Status */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Model Health</h2>
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${statusCfg.bg}`} />
          <span className={`text-sm font-medium ${statusCfg.color}`}>
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
          value={data.avg_confidence ? `${(data.avg_confidence * 100).toFixed(1)}%` : "N/A"}
          color={
            data.avg_confidence && data.avg_confidence >= 0.9
              ? "text-emerald-400"
              : data.avg_confidence && data.avg_confidence >= 0.7
              ? "text-yellow-400"
              : "text-red-400"
          }
        />
        <MiniCard
          label="Accuracy"
          value={data.accuracy ? `${(data.accuracy * 100).toFixed(1)}%` : "N/A"}
          color={
            data.accuracy && data.accuracy >= 0.8
              ? "text-emerald-400"
              : data.accuracy && data.accuracy >= 0.6
              ? "text-yellow-400"
              : "text-red-400"
          }
        />
        <MiniCard
          label="Suppression Rate"
          value={`${data.pct_suppressed.toFixed(1)}%`}
          color="text-blue-400"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Confidence Distribution */}
        <div>
          <p className="text-xs text-gray-400 mb-2 font-medium">Confidence Distribution</p>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={confData} barSize={40}>
              <XAxis
                dataKey="name"
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1f2937",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f3f4f6",
                  fontSize: "12px",
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {confData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Prediction Split */}
        <div>
          <p className="text-xs text-gray-400 mb-2 font-medium">Prediction Split</p>
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
                    className="h-2.5 w-2.5 rounded-sm"
                    style={{ backgroundColor: item.fill }}
                  />
                  <span className="text-xs text-gray-400">
                    {item.name}: <span className="text-white font-medium">{item.value}%</span>
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

function MiniCard({
  label,
  value,
  color = "text-white",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="bg-gray-900/50 rounded-lg px-3 py-2.5">
      <p className="text-gray-500 text-[10px] uppercase tracking-wide">{label}</p>
      <p className={`text-lg font-bold ${color}`}>{value}</p>
    </div>
  );
}
