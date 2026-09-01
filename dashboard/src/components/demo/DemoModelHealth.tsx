/**
 * Demo-mode model health: cumulative shift aggregates from the replay engine.
 * Compact card in the rail; full view in a drawer, mirroring ModelHealth.
 */

import ModelHealthView from "../ModelHealthView";
import CompactModelHealthCard from "../CompactModelHealthCard";
import DrawerShell from "../DrawerShell";
import { IconX } from "../icons";
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
            <p className="text-[10px] text-ink-low">
              Full breakdown for the current shift.
            </p>
            <button
              onClick={onClose}
              className="text-ink-low hover:text-ink px-1 ml-3"
              aria-label="Close model health details"
            >
              <IconX size={14} />
            </button>
          </div>
          <ModelHealthView data={snapshot.modelHealth} animate={false} />
        </div>
      </DrawerShell>
    </>
  );
}
