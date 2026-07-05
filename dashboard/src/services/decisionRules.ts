/**
 * Demo-mode decision rule table.
 *
 * Single source of truth for the heuristic verdicts shown in demo mode:
 * demoPredict() derives its prediction from evaluateRules(), and the alert
 * detail drawer reuses the same factors as "context signals".
 *
 * Rules per notification type are ordered — the first `when` match wins,
 * and every list ends with a catch-all.
 */

import type { NotificationPayload } from "./api";

export interface DecisionFactor {
  label: string;
  detail: string;
  direction: "flag" | "suppress" | "context";
}

export interface RuleDecision {
  needsIntervention: boolean;
  confidenceRange: [number, number];
  factors: DecisionFactor[];
}

type Rng = () => number;

interface Rule {
  when: (p: NotificationPayload) => boolean;
  needsIntervention: boolean | ((p: NotificationPayload, rng: Rng) => boolean);
  confidenceRange: [number, number];
  factors: (p: NotificationPayload) => DecisionFactor[];
}

const STUCK_RULES: Rule[] = [
  {
    when: (p) => p.traffic_condition === "heavy" || p.traffic_condition === "standstill",
    needsIntervention: false,
    confidenceRange: [0.8, 0.92],
    factors: (p) => [
      {
        label: "Heavy traffic",
        detail: `Stopped in ${p.traffic_condition} traffic — vehicles typically clear on their own`,
        direction: "suppress",
      },
    ],
  },
  {
    when: (p) => p.construction_zone !== "none" && Boolean(p.construction_zone),
    needsIntervention: false,
    confidenceRange: [0.75, 0.88],
    factors: (p) => [
      {
        label: "Construction zone",
        detail: `Active ${p.construction_zone} construction legitimately slows or halts traffic`,
        direction: "suppress",
      },
    ],
  },
  {
    when: (p) => p.traffic_condition === "light",
    needsIntervention: true,
    confidenceRange: [0.7, 0.85],
    factors: () => [
      {
        label: "Clear road",
        detail: "Stopped despite light traffic and no construction — likely needs help",
        direction: "flag",
      },
    ],
  },
  {
    when: () => true,
    needsIntervention: false,
    confidenceRange: [0.6, 0.75],
    factors: (p) => [
      {
        label: "Moderate traffic",
        detail: `Ambiguous context in ${p.traffic_condition} traffic — slight lean to self-recovery`,
        direction: "suppress",
      },
    ],
  },
];

const VERIFICATION_RULES: Rule[] = [
  {
    // Obstruction confirmed is the strongest signal — checked before pedestrian context
    when: (p) => p.notification_subtype === "object_query" && p.object_in_path === true,
    needsIntervention: true,
    confidenceRange: [0.85, 0.95],
    factors: () => [
      {
        label: "Obstruction confirmed",
        detail: "An object is actually in the vehicle's path — intervention required",
        direction: "flag",
      },
    ],
  },
  {
    when: (p) => p.notification_subtype === "object_query" && p.speed > 10,
    needsIntervention: true,
    confidenceRange: [0.85, 0.95],
    factors: (p) => [
      {
        label: "Querying while moving",
        detail: `Object query at ${p.speed} mph is unusual — likely a real obstruction`,
        direction: "flag",
      },
    ],
  },
  {
    when: (p) => p.notification_subtype === "object_query" && p.pedestrian_density > 0.5,
    needsIntervention: false,
    confidenceRange: [0.8, 0.92],
    factors: (p) => [
      {
        label: "High pedestrian area",
        detail: `Pedestrian density ${p.pedestrian_density} — object queries here are usually passers-by`,
        direction: "suppress",
      },
    ],
  },
  {
    when: (p) => p.notification_subtype === "object_query" && p.pedestrian_density <= 0.3,
    needsIntervention: true,
    confidenceRange: [0.75, 0.88],
    factors: (p) => [
      {
        label: "Low pedestrian area",
        detail: `Pedestrian density ${p.pedestrian_density} — little foot traffic to explain the query`,
        direction: "flag",
      },
    ],
  },
  {
    when: (p) => p.notification_subtype === "object_query",
    needsIntervention: false,
    confidenceRange: [0.65, 0.8],
    factors: () => [
      {
        label: "Ambiguous pedestrian context",
        detail: "Mid-range pedestrian density — weak lean to false positive",
        direction: "suppress",
      },
    ],
  },
  {
    when: (p) => p.notification_subtype === "traffic_signal_verify",
    needsIntervention: true,
    confidenceRange: [0.7, 0.8],
    factors: () => [
      {
        label: "Signal verification",
        detail: "Signal-state mismatches are usually real mapping or perception issues",
        direction: "flag",
      },
    ],
  },
  {
    when: (p) => p.notification_subtype === "lane_mapping_verify",
    needsIntervention: (_p, rng) => rng() > 0.7,
    confidenceRange: [0.65, 0.78],
    factors: () => [
      {
        label: "Lane mapping ambiguity",
        detail: "Lane geometry disagreement — verdict depends on surrounding context",
        direction: "context",
      },
    ],
  },
  {
    when: () => true,
    needsIntervention: true,
    confidenceRange: [0.6, 0.75],
    factors: () => [
      {
        label: "Unclassified verification",
        detail: "Unknown verification subtype — flagged for review",
        direction: "flag",
      },
    ],
  },
];

const EV_RULES: Rule[] = [
  {
    when: (p) => (p.ev_distance ?? 999) > 200,
    needsIntervention: false,
    confidenceRange: [0.85, 0.95],
    factors: (p) => [
      {
        label: "EV far away",
        detail: `Emergency vehicle ${p.ev_distance ?? "500+"} m out — no yield action needed yet`,
        direction: "suppress",
      },
    ],
  },
  {
    when: (p) => (p.ev_distance ?? 999) < 50,
    needsIntervention: true,
    confidenceRange: [0.8, 0.92],
    factors: (p) => [
      {
        label: "EV close",
        detail: `Emergency vehicle within ${p.ev_distance} m — vehicle must yield now`,
        direction: "flag",
      },
    ],
  },
  {
    when: () => true,
    needsIntervention: false,
    confidenceRange: [0.6, 0.75],
    factors: (p) => [
      {
        label: "EV mid-distance",
        detail: `Emergency vehicle at ${p.ev_distance} m — usually resolves without intervention`,
        direction: "suppress",
      },
    ],
  },
];

const SPEED_ANOMALY_RULES: Rule[] = [
  {
    when: (p) => p.traffic_condition === "heavy" || p.traffic_condition === "standstill",
    needsIntervention: false,
    confidenceRange: [0.8, 0.9],
    factors: (p) => [
      {
        label: "Traffic explains speed",
        detail: `Below expected speed in ${p.traffic_condition} traffic — expected behavior`,
        direction: "suppress",
      },
    ],
  },
  {
    when: (p) => p.traffic_condition === "light",
    needsIntervention: true,
    confidenceRange: [0.75, 0.85],
    factors: (p) => [
      {
        label: "Slow on clear road",
        detail: `${p.speed} mph where ${p.expected_speed} expected, with light traffic`,
        direction: "flag",
      },
    ],
  },
  {
    when: () => true,
    needsIntervention: false,
    confidenceRange: [0.65, 0.8],
    factors: () => [
      {
        label: "Moderate traffic",
        detail: "Speed gap partially explained by traffic — weak lean to false positive",
        direction: "suppress",
      },
    ],
  },
];

const IMPACT_RULES: Rule[] = [
  {
    when: (p) => p.road_type === "residential" || p.road_type === "downtown",
    needsIntervention: false,
    confidenceRange: [0.6, 0.75],
    factors: (p) => [
      {
        label: "Rough road surface",
        detail: `Low-level impacts on ${p.road_type} streets are usually speed bumps or potholes`,
        direction: "suppress",
      },
    ],
  },
  {
    when: () => true,
    needsIntervention: true,
    confidenceRange: [0.7, 0.85],
    factors: (p) => [
      {
        label: "Impact on smooth road",
        detail: `Impact detected on ${p.road_type} — no surface features to explain it`,
        direction: "flag",
      },
    ],
  },
];

const PASSENGER_ASSIST_RULES: Rule[] = [
  {
    when: () => true,
    needsIntervention: true,
    confidenceRange: [0.95, 0.99],
    factors: () => [
      {
        label: "Rider request",
        detail: "A passenger asked for help — always routed to an operator",
        direction: "flag",
      },
    ],
  },
];

const FALLBACK_RULES: Rule[] = [
  {
    when: () => true,
    needsIntervention: true,
    confidenceRange: [0.5, 0.7],
    factors: () => [
      {
        label: "Unknown notification type",
        detail: "No rule coverage — flagged for review",
        direction: "flag",
      },
    ],
  },
];

const RULES: Record<string, Rule[]> = {
  stuck: STUCK_RULES,
  verification_request: VERIFICATION_RULES,
  emergency_vehicle_alert: EV_RULES,
  speed_anomaly: SPEED_ANOMALY_RULES,
  impact_l0: IMPACT_RULES,
  passenger_assist: PASSENGER_ASSIST_RULES,
};

/** Payload-derived context signals appended to every decision. */
function contextFactors(p: NotificationPayload): DecisionFactor[] {
  const factors: DecisionFactor[] = [];
  if (p.speed < 5 && p.time_since_stop > 0) {
    factors.push({
      label: "Stationary",
      detail: `Stopped for ${Math.round(p.time_since_stop)}s`,
      direction: "context",
    });
  }
  return factors;
}

export function evaluateRules(
  payload: NotificationPayload,
  rng: Rng = Math.random
): RuleDecision {
  const rules = RULES[payload.notification_type] ?? FALLBACK_RULES;
  // Safe: every rule list ends with a catch-all `when: () => true`
  const rule = rules.find((r) => r.when(payload)) as Rule;

  const needsIntervention =
    typeof rule.needsIntervention === "function"
      ? rule.needsIntervention(payload, rng)
      : rule.needsIntervention;

  return {
    needsIntervention,
    confidenceRange: rule.confidenceRange,
    factors: [...rule.factors(payload), ...contextFactors(payload)],
  };
}
