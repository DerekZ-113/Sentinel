/**
 * Demo-mode model health: cumulative shift aggregates from the replay engine.
 */

import ModelHealthView from "../ModelHealthView";
import { getEngine } from "../../demo/engineInstance";
import { useEngineSnapshot } from "../../demo/useEngine";
import type { ReplayEngine } from "../../demo/types";

export default function DemoModelHealth({
  engine = getEngine(),
}: {
  engine?: ReplayEngine;
}) {
  const snapshot = useEngineSnapshot(engine);
  return <ModelHealthView data={snapshot.modelHealth} animate={false} />;
}
