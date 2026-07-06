/**
 * Demo-mode alert detail drawer: model decision visualization, context
 * signals from the shared rule table, telemetry, sector map, and the
 * vehicle's recent history from the replay ring.
 */

import { getEngine } from "../demo/engineInstance";
import { useEngineSnapshot, useNow } from "../demo/useEngine";
import { relativeTime } from "../demo/format";
import type { DemoAlert, ReplayEngine } from "../demo/types";
import { evaluateRules } from "../services/decisionRules";
import type { NotificationPayload } from "../services/api";
import { TypeBadge } from "./alertRowParts";
import SectorMap from "./SectorMap";
import DrawerShell from "./DrawerShell";

const THRESHOLD = 0.5;
const HISTORY_SIZE = 6;

/** Reconstruct a rule-table payload from an alert record (nulls defaulted). */
function payloadFromAlert(alert: DemoAlert): NotificationPayload {
  return {
    vehicle_id: alert.vehicle_id,
    speed: alert.speed ?? 0,
    expected_speed: alert.expected_speed ?? alert.speed ?? 0,
    road_type: alert.road_type ?? "main_road",
    traffic_condition: alert.traffic_condition ?? "moderate",
    construction_zone: alert.construction_zone ?? "none",
    notification_type: alert.notification_type,
    notification_subtype: alert.notification_subtype,
    ev_distance: alert.ev_distance ?? null,
    pedestrian_density: alert.pedestrian_density ?? 0,
    object_in_path: alert.object_in_path ?? false,
    time_since_stop: alert.time_since_stop ?? 0,
  };
}

const FACTOR_STYLES: Record<string, string> = {
  flag: "bg-red-900/30 border-red-800/50 text-red-300",
  suppress: "bg-emerald-900/30 border-emerald-800/50 text-emerald-300",
  context: "bg-gray-800/60 border-gray-700/50 text-gray-300",
};

interface AlertDetailDrawerProps {
  alert: DemoAlert | null;
  onClose: () => void;
  onSelect: (alert: DemoAlert) => void;
  engine?: ReplayEngine;
}

export default function AlertDetailDrawer({
  alert,
  onClose,
  onSelect,
  engine = getEngine(),
}: AlertDetailDrawerProps) {
  const snapshot = useEngineSnapshot(engine);
  const now = useNow(1000);

  if (alert === null) return null;

  const rawScore =
    alert.raw_score ??
    (alert.needs_intervention_predicted ? alert.confidence : 1 - alert.confidence);
  const flagged = alert.needs_intervention_predicted;
  const factors = evaluateRules(payloadFromAlert(alert)).factors;
  const history = snapshot.events
    .filter((e) => e.vehicle_id === alert.vehicle_id && e.id !== alert.id)
    .slice(0, HISTORY_SIZE);

  const correct =
    alert.needs_intervention_actual !== null
      ? alert.needs_intervention_predicted === alert.needs_intervention_actual
      : null;

  return (
    <DrawerShell
      open
      onClose={onClose}
      ariaLabel={`Alert detail for ${alert.vehicle_id}`}
    >
      <div className="p-5 space-y-5">
          {/* Header */}
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-lg font-semibold text-white font-mono">
                  {alert.vehicle_id}
                </h2>
                {alert.source === "manual" && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-900/60 text-blue-300 border border-blue-800/50">
                    manual
                  </span>
                )}
              </div>
              <div className="mt-1.5 flex items-center gap-2 flex-wrap">
                <TypeBadge
                  type={alert.notification_type}
                  subtype={alert.notification_subtype}
                />
                <span className="text-xs text-gray-500 tabular-nums">
                  {relativeTime(now, alert.time)}
                </span>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-white text-lg leading-none px-1"
              aria-label="Close detail"
            >
              ✕
            </button>
          </div>

          {/* Verdict */}
          <div
            className={`rounded-lg border p-3 ${
              flagged
                ? "bg-red-900/20 border-red-800/50"
                : "bg-emerald-900/20 border-emerald-800/50"
            }`}
          >
            <div className="flex items-center justify-between">
              <p
                className={`font-bold ${flagged ? "text-red-400" : "text-emerald-400"}`}
              >
                {flagged ? "⚠ Flagged for review" : "✓ Suppressed"}
              </p>
              {alert.needs_intervention_actual !== null && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full border ${
                    correct
                      ? "bg-emerald-900/40 border-emerald-800/50 text-emerald-300"
                      : "bg-red-900/40 border-red-800/50 text-red-300"
                  }`}
                >
                  {alert.needs_intervention_actual
                    ? "Ground truth: real"
                    : "Ground truth: false positive"}{" "}
                  · {correct ? "correct" : "missed"}
                </span>
              )}
            </div>

            {/* Score vs threshold */}
            <div className="mt-3">
              <div className="flex justify-between text-[10px] text-gray-500 mb-1">
                <span>
                  Model score:{" "}
                  <span className="text-gray-300 font-mono">{rawScore.toFixed(3)}</span>
                </span>
                <span>threshold {THRESHOLD}</span>
              </div>
              <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${flagged ? "bg-red-500" : "bg-emerald-500"}`}
                  style={{ width: `${Math.min(100, rawScore * 100)}%` }}
                />
                <div
                  className="absolute top-0 h-full w-px bg-gray-400"
                  style={{ left: `${THRESHOLD * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Context signals */}
          <div>
            <p className="text-xs text-gray-400 font-medium mb-2">Context signals</p>
            <div className="space-y-2">
              {factors.map((factor) => (
                <div
                  key={factor.label}
                  className={`rounded-lg border px-3 py-2 ${FACTOR_STYLES[factor.direction]}`}
                >
                  <p className="text-xs font-semibold">{factor.label}</p>
                  <p className="text-[11px] opacity-80 mt-0.5">{factor.detail}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Location */}
          <div>
            <p className="text-xs text-gray-400 font-medium mb-2">Location</p>
            <SectorMap
              latitude={alert.latitude}
              longitude={alert.longitude}
              seed={alert.vehicle_id}
            />
          </div>

          {/* Telemetry */}
          <div>
            <p className="text-xs text-gray-400 font-medium mb-2">Telemetry</p>
            <div className="grid grid-cols-2 gap-2">
              <TelemetryCell label="Speed" value={fmtNumber(alert.speed, " mph")} />
              <TelemetryCell
                label="Expected speed"
                value={fmtNumber(alert.expected_speed, " mph")}
              />
              <TelemetryCell label="Road type" value={fmtText(alert.road_type)} />
              <TelemetryCell label="Traffic" value={fmtText(alert.traffic_condition)} />
              <TelemetryCell
                label="Construction"
                value={fmtText(alert.construction_zone)}
              />
              <TelemetryCell
                label="Pedestrian density"
                value={fmtNumber(alert.pedestrian_density, "")}
              />
              {alert.ev_distance !== null && alert.ev_distance !== undefined && (
                <TelemetryCell label="EV distance" value={fmtNumber(alert.ev_distance, " m")} />
              )}
              <TelemetryCell
                label="Time since stop"
                value={fmtNumber(alert.time_since_stop, " s")}
              />
              <TelemetryCell
                label="Object in path"
                value={
                  alert.object_in_path === null || alert.object_in_path === undefined
                    ? "—"
                    : alert.object_in_path
                    ? "yes"
                    : "no"
                }
              />
              <TelemetryCell
                label="Confidence"
                value={`${(alert.confidence * 100).toFixed(1)}%`}
              />
            </div>
          </div>

          {/* Vehicle history */}
          <div>
            <p className="text-xs text-gray-400 font-medium mb-2">
              Recent activity — {alert.vehicle_id}
            </p>
            {history.length === 0 ? (
              <p className="text-xs text-gray-600">
                No other alerts from this vehicle in the current window.
              </p>
            ) : (
              <ul className="space-y-1">
                {history.map((event) => (
                  <li key={event.id}>
                    <button
                      onClick={() => onSelect(event)}
                      className="w-full flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-gray-800/60 transition-colors text-left"
                    >
                      <span className="text-[11px] text-gray-500 tabular-nums w-16 shrink-0">
                        {relativeTime(now, event.time)}
                      </span>
                      <TypeBadge
                        type={event.notification_type}
                        subtype={event.notification_subtype}
                      />
                      <span
                        className={`ml-auto text-xs font-medium ${
                          event.needs_intervention_predicted
                            ? "text-red-400"
                            : "text-emerald-400"
                        }`}
                      >
                        {event.needs_intervention_predicted ? "⚠" : "✓"}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
      </div>
    </DrawerShell>
  );
}

function TelemetryCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800/50 rounded-lg px-3 py-2">
      <p className="text-gray-500 text-[10px] uppercase tracking-wide">{label}</p>
      <p className="text-sm text-gray-200 font-medium">{value}</p>
    </div>
  );
}

function fmtNumber(value: number | null | undefined, unit: string): string {
  if (value === null || value === undefined) return "—";
  return `${value}${unit}`;
}

function fmtText(value: string | null | undefined): string {
  if (!value) return "—";
  return value.replace(/_/g, " ");
}
