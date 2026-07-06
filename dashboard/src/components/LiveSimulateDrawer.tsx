/**
 * Live-mode Simulate slide-over. Wraps SimulatePanel; predictions run
 * against the real API. No demo imports — safe in live bundles.
 */

import SimulatePanel from "./SimulatePanel";
import DrawerShell from "./DrawerShell";

interface LiveSimulateDrawerProps {
  open: boolean;
  onClose: () => void;
}

export default function LiveSimulateDrawer({
  open,
  onClose,
}: LiveSimulateDrawerProps) {
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
            Run a notification payload against the model API.
          </p>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-white text-lg leading-none px-1 ml-3"
            aria-label="Close simulate"
          >
            ✕
          </button>
        </div>
        <SimulatePanel />
      </div>
    </DrawerShell>
  );
}
