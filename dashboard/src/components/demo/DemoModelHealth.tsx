/**
 * Demo-mode model health: cumulative shift aggregates from the replay engine.
 * Compact card in the rail; full view in a drawer, mirroring ModelHealth.
 */

import ModelHealthView from "../ModelHealthView";
import CompactModelHealthCard from "../CompactModelHealthCard";
import DrawerShell from "../DrawerShell";
import { getEngine } from "../../demo/engineInstance";
import { useEngineSnapshot } from "../../demo/useEngine";
import type { ReplayEngine } from "../../demo/types";

export default function DemoModelHealth({
  engine = getEngine(),
  expanded = false,
  onExpand,
  onClose,
}: {
  engine?: ReplayEngine;
  expanded?: boolean;
  onExpand?: () => void;
  onClose?: () => void;
}) {
  const snapshot = useEngineSnapshot(engine);

  return (
    <>
      <CompactModelHealthCard data={snapshot.modelHealth} onExpand={onExpand} />
      <DrawerShell
        open={expanded}
        onClose={onClose ?? (() => {})}
        ariaLabel="Model health details"
        widthClassName="w-full sm:max-w-xl"
      >
        <div className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-xs text-gray-500">
              Full breakdown for the current shift.
            </p>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-white text-lg leading-none px-1 ml-3"
              aria-label="Close model health details"
            >
              ✕
            </button>
          </div>
          <ModelHealthView data={snapshot.modelHealth} animate={false} />
        </div>
      </DrawerShell>
    </>
  );
}
