/**
 * Demo-mode Simulate slide-over. Wraps SimulatePanel; each successful
 * prediction is injected into the live replay stream as a manual alert.
 */

import SimulatePanel from "./SimulatePanel";
import DrawerShell from "./DrawerShell";
import { IconX } from "./icons";
import { getEngine } from "../demo/engineInstance";
import { buildAlertFromPrediction } from "../demo/replayEngine";
import type { ReplayEngine } from "../demo/types";

interface SimulateDrawerProps {
  open: boolean;
  onClose: () => void;
  engine?: ReplayEngine;
}

export default function SimulateDrawer({
  open,
  onClose,
  engine = getEngine(),
}: SimulateDrawerProps) {
  return (
    <DrawerShell
      open={open}
      onClose={onClose}
      ariaLabel="Simulate notification"
      widthClassName="w-full sm:max-w-xl"
    >
      <div className="p-5 space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-[10px] text-ink-low">
            Predictions run against the demo rule model and enter the live
            stream as{" "}
            <span className="text-accent">manual</span> alerts.
          </p>
          <button
            onClick={onClose}
            className="text-ink-low hover:text-ink px-1 ml-3"
            aria-label="Close simulate"
          >
            <IconX size={14} />
          </button>
        </div>
        <SimulatePanel
          onResult={(result, payload) =>
            engine.injectManual(buildAlertFromPrediction(result, payload))
          }
        />
      </div>
    </DrawerShell>
  );
}
