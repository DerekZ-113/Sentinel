/**
 * Demo-mode Simulate slide-over. Wraps SimulatePanel; each successful
 * prediction is injected into the live replay stream as a manual alert.
 */

import SimulatePanel from "./SimulatePanel";
import DrawerShell from "./DrawerShell";
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
          <p className="text-xs text-gray-500">
            Predictions run against the demo rule model and enter the live
            stream as{" "}
            <span className="text-blue-300">manual</span> alerts.
          </p>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-white text-lg leading-none px-1 ml-3"
            aria-label="Close simulate"
          >
            ✕
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
