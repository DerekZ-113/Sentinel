/**
 * Demo-mode Simulate slide-over. Wraps SimulatePanel; each successful
 * prediction is injected into the live replay stream as a manual alert.
 */

import { useEffect } from "react";
import SimulatePanel from "./SimulatePanel";
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
  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer */}
      <aside
        className="fixed inset-y-0 right-0 w-full sm:max-w-xl z-50 bg-gray-900 border-l border-gray-800 overflow-y-auto"
        role="dialog"
        aria-label="Simulate notification"
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
      </aside>
    </>
  );
}
