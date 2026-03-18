import { useState } from "react";

export default function DemoBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (import.meta.env.VITE_DEMO_MODE !== "true" || dismissed) return null;

  return (
    <div className="bg-blue-900/30 border border-blue-800/40 rounded-lg px-4 py-2.5 flex items-center justify-between text-sm">
      <p className="text-blue-300/80 text-xs">
        Demo mode — showing pre-seeded data.{" "}
        <a
          href="https://github.com/DerekZ-113/Sentinel"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-400 hover:text-blue-300 underline"
        >
          Run with Docker
        </a>{" "}
        for live predictions.
      </p>
      <button
        onClick={() => setDismissed(true)}
        className="text-blue-400/60 hover:text-blue-300 text-xs ml-4"
        aria-label="Dismiss"
      >
        x
      </button>
    </div>
  );
}
