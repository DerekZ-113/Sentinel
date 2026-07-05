/**
 * Demo-mode FP chart: rolling last-30-min buckets from the replay engine.
 */

import { useMemo } from "react";
import FPRateChartView from "../FPRateChartView";
import { getEngine } from "../../demo/engineInstance";
import { useEngineSnapshot } from "../../demo/useEngine";
import { minuteLabel } from "../../demo/format";
import type { ReplayEngine } from "../../demo/types";

export default function DemoFPRateChart({
  engine = getEngine(),
}: {
  engine?: ReplayEngine;
}) {
  const snapshot = useEngineSnapshot(engine);
  const data = useMemo(
    () => ({ time_window_hours: 0.5, buckets: snapshot.fpBuckets }),
    [snapshot.fpBuckets]
  );

  return (
    <FPRateChartView
      data={data}
      windowLabel="Last 30 min · live"
      tickFormatter={minuteLabel}
      animate={false}
    />
  );
}
