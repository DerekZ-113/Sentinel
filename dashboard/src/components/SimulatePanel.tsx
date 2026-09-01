import { useState } from "react";
import { postPredict } from "../services/api";
import type { NotificationPayload, PredictionResponse } from "../services/api";
import { VerdictChip } from "./alertRowParts";

const NOTIFICATION_TYPES = [
  "stuck",
  "verification_request",
  "emergency_vehicle_alert",
  "speed_anomaly",
  "impact_l0",
  "passenger_assist",
];

const SUBTYPES: Record<string, string[]> = {
  verification_request: ["object_query", "traffic_signal_verify", "lane_mapping_verify"],
};

const ROAD_TYPES = ["highway", "main_road", "residential", "downtown", "school_zone"];
const TRAFFIC_CONDITIONS = ["light", "moderate", "heavy", "standstill"];
const CONSTRUCTION_ZONES = ["none", "temporary", "persistent", "flagger"];

interface SimulatePanelProps {
  /** Called with each successful prediction (demo mode injects it into the stream). */
  onResult?: (result: PredictionResponse, payload: NotificationPayload) => void;
}

export default function SimulatePanel({ onResult }: SimulatePanelProps) {
  const [notifType, setNotifType] = useState("stuck");
  const [subtype, setSubtype] = useState<string | null>(null);
  const [roadType, setRoadType] = useState("downtown");
  const [traffic, setTraffic] = useState("heavy");
  const [construction, setConstruction] = useState("none");
  const [speed, setSpeed] = useState(0);
  const [expectedSpeed, setExpectedSpeed] = useState(35);
  const [pedestrianDensity, setPedestrianDensity] = useState(0.3);
  const [evDistance, setEvDistance] = useState(200);
  const [timeSinceStop, setTimeSinceStop] = useState(120);
  const [objectInPath, setObjectInPath] = useState(false);

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const showSubtype = SUBTYPES[notifType] !== undefined;
  const showEv = notifType === "emergency_vehicle_alert";
  const showObject = notifType === "verification_request";

  async function handlePredict() {
    setLoading(true);
    setResult(null);
    setError(null);
    const payload: NotificationPayload = {
      vehicle_id: "sim_" + Math.random().toString(36).slice(2, 6),
      speed,
      expected_speed: expectedSpeed,
      road_type: roadType,
      traffic_condition: traffic,
      construction_zone: construction,
      notification_type: notifType,
      notification_subtype: showSubtype ? subtype : null,
      ev_distance: showEv ? evDistance : null,
      pedestrian_density: pedestrianDensity,
      object_in_path: showObject ? objectInPath : false,
      time_since_stop: timeSinceStop,
    };
    try {
      const res = await postPredict(payload);
      setResult(res);
      onResult?.(res, payload);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    setLoading(false);
  }

  return (
    <div className="bg-panel border border-hairline rounded-xs p-5">
      <h2 className="text-[11px] font-semibold uppercase tracking-[0.1em] text-ink-mid mb-4">
        Simulate Notification
      </h2>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-5">
        {/* Notification Type */}
        <SelectField
          label="Notification Type"
          value={notifType}
          options={NOTIFICATION_TYPES}
          onChange={(v) => {
            setNotifType(v);
            setSubtype(SUBTYPES[v]?.[0] || null);
          }}
        />

        {/* Subtype */}
        {showSubtype && (
          <SelectField
            label="Subtype"
            value={subtype || ""}
            options={SUBTYPES[notifType]}
            onChange={setSubtype}
          />
        )}

        {/* Road Type */}
        <SelectField label="Road Type" value={roadType} options={ROAD_TYPES} onChange={setRoadType} />

        {/* Traffic */}
        <SelectField label="Traffic" value={traffic} options={TRAFFIC_CONDITIONS} onChange={setTraffic} />

        {/* Construction */}
        <SelectField label="Construction" value={construction} options={CONSTRUCTION_ZONES} onChange={setConstruction} />

        {/* Speed */}
        <SliderField label="Speed" value={speed} min={0} max={80} unit="mph" onChange={setSpeed} />

        {/* Expected Speed */}
        <SliderField label="Expected Speed" value={expectedSpeed} min={0} max={80} unit="mph" onChange={setExpectedSpeed} />

        {/* Pedestrian Density */}
        <SliderField
          label="Pedestrian Density"
          value={pedestrianDensity}
          min={0}
          max={1}
          step={0.05}
          onChange={setPedestrianDensity}
        />

        {/* Time Since Stop */}
        <SliderField label="Time Since Stop" value={timeSinceStop} min={0} max={600} unit="s" onChange={setTimeSinceStop} />

        {/* EV Distance */}
        {showEv && (
          <SliderField label="EV Distance" value={evDistance} min={0} max={500} unit="m" onChange={setEvDistance} />
        )}

        {/* Object in Path */}
        {showObject && (
          <div>
            <label className="block text-[10px] uppercase tracking-[0.1em] text-ink-micro mb-1">
              Object in Path
            </label>
            <button
              onClick={() => setObjectInPath(!objectInPath)}
              className={`px-3 py-2 rounded-xs text-[11px] uppercase tracking-[0.05em] font-medium border transition-colors w-full ${
                objectInPath
                  ? "bg-crit/14 border-crit/40 text-crit"
                  : "bg-inset border-hairline-2 text-ink-mid"
              }`}
            >
              {objectInPath ? "Yes — Obstruction" : "No — Clear"}
            </button>
          </div>
        )}
      </div>

      {/* Predict Button */}
      <button
        onClick={handlePredict}
        disabled={loading}
        className="border border-accent/60 text-accent hover:bg-accent/10 disabled:border-hairline-2 disabled:text-ink-low disabled:hover:bg-transparent rounded-xs px-6 py-2.5 text-[11px] font-medium uppercase tracking-[0.1em] transition-colors"
      >
        {loading ? "Predicting..." : "Run Prediction"}
      </button>

      {/* Error */}
      {error && (
        <div className="mt-5 p-4 rounded-xs border bg-crit/10 border-crit/40">
          <p className="text-crit text-sm">Prediction failed: {error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div
          className={`mt-5 p-4 rounded-xs border ${
            result.needs_intervention
              ? "bg-crit/10 border-crit/40"
              : "bg-ok/10 border-ok/40"
          }`}
        >
          <div className="flex items-center gap-3 mb-2">
            <VerdictChip flagged={result.needs_intervention} />
            <div>
              <p
                className={`text-[13px] font-semibold ${
                  result.needs_intervention ? "text-crit" : "text-ok"
                }`}
              >
                {result.needs_intervention
                  ? "Flag — Intervention Needed"
                  : "Suppress — Likely False Positive"}
              </p>
              <p className="text-ink-mid text-[11px] tabular-nums">
                Confidence: {(result.confidence * 100).toFixed(1)}% •
                Raw score: {result.raw_score.toFixed(4)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Reusable field components
// ============================================================================

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.1em] text-ink-micro mb-1">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-inset border border-hairline-2 rounded-xs px-3 py-2
                   text-[12px] text-ink-data focus:outline-none focus:border-accent"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt.replace(/_/g, " ")}
          </option>
        ))}
      </select>
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step = 1,
  unit = "",
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="block text-[10px] uppercase tracking-[0.1em] text-ink-micro mb-1">
        {label}: <span className="text-ink font-medium tabular-nums">{value}{unit && ` ${unit}`}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent"
      />
    </div>
  );
}
