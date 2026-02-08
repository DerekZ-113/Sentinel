import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TypeStats } from "../services/api";

interface TypeBreakdownProps {
  byType: TypeStats[];
}

const TYPE_LABELS: Record<string, string> = {
  verification_request: "Verification",
  emergency_vehicle_alert: "EV Alert",
  stuck: "Stuck",
  speed_anomaly: "Speed Anomaly",
  impact_l0: "Impact",
  passenger_assist: "Passenger",
};

export default function TypeBreakdown({ byType }: TypeBreakdownProps) {
  const data = byType.map((t) => ({
    name: TYPE_LABELS[t.notification_type] || t.notification_type,
    Flagged: t.flagged,
    Suppressed: t.suppressed,
    total: t.total,
  }));

  return (
    <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-5 h-[420px] flex flex-col">
      <h2 className="text-lg font-semibold text-white mb-4">
        Alerts by Notification Type
      </h2>

      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} barGap={2}>
          <XAxis
            dataKey="name"
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#9ca3af", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              color: "#f3f4f6",
              fontSize: "13px",
            }}
            cursor={{ fill: "rgba(255,255,255,0.03)" }}
          />
          <Legend
            wrapperStyle={{ fontSize: "13px", color: "#9ca3af" }}
          />
          <Bar dataKey="Suppressed" stackId="a" fill="#10b981" radius={[0, 0, 0, 0]} />
          <Bar dataKey="Flagged" stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
