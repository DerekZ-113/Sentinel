import { useEffect, useState } from "react";
import { fetchModelHealth } from "../services/api";
import type { ModelHealthResponse } from "../services/api";
import ModelHealthView from "./ModelHealthView";

export default function ModelHealth() {
  const [data, setData] = useState<ModelHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelHealth()
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  function retry() {
    setError(null);
    setData(null);
    fetchModelHealth()
      .then(setData)
      .catch((err) => setError(err.message));
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-800/50 rounded-xl p-6 flex flex-col items-center justify-center gap-3 h-64">
        <p className="text-red-400 text-sm">{error}</p>
        <button
          onClick={retry}
          className="text-xs text-red-300 hover:text-white border border-red-700 px-3 py-1 rounded-lg transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-gray-800/40 border border-gray-700/50 rounded-xl p-6 flex items-center justify-center text-gray-500 h-64">
        Loading model health...
      </div>
    );
  }

  return <ModelHealthView data={data} />;
}
