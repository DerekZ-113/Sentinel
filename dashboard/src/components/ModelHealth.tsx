import { useEffect, useRef, useState } from "react";
import { fetchModelHealth } from "../services/api";
import type { ModelHealthResponse } from "../services/api";
import ModelHealthView from "./ModelHealthView";
import CompactModelHealthCard from "./CompactModelHealthCard";
import DrawerShell from "./DrawerShell";

interface ModelHealthProps {
  refreshToken?: number;
  expanded?: boolean;
  onExpand?: () => void;
  onClose?: () => void;
}

export default function ModelHealth({
  refreshToken = 0,
  expanded = false,
  onExpand,
  onClose,
}: ModelHealthProps) {
  const [data, setData] = useState<ModelHealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const hasLoadedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const isRefresh = hasLoadedRef.current;
    fetchModelHealth()
      .then((nextData) => {
        if (cancelled) return;
        setData(nextData);
        setError(null);
        setRefreshError(null);
        hasLoadedRef.current = true;
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Unknown error";
        if (isRefresh) {
          setRefreshError(`Refresh failed: ${message}`);
        } else {
          setError(message);
        }
      });
    return () => { cancelled = true; };
  }, [refreshToken]);

  function retry() {
    setError(null);
    setRefreshError(null);
    setData(null);
    hasLoadedRef.current = false;
    fetchModelHealth()
      .then((nextData) => {
        setData(nextData);
        setError(null);
        hasLoadedRef.current = true;
      })
      .catch((err) => {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
      });
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

  return (
    <div className="space-y-3">
      {refreshError && (
        <p className="rounded border border-yellow-700/40 bg-yellow-900/10 px-3 py-1.5 text-xs text-yellow-300/80">
          {refreshError}
        </p>
      )}
      <CompactModelHealthCard data={data} onExpand={onExpand} />
      <DrawerShell
        open={expanded}
        onClose={onClose ?? (() => {})}
        ariaLabel="Model health details"
        widthClassName="w-full sm:max-w-xl"
      >
        <div className="p-5 space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-xs text-gray-500">
              Full breakdown for the current window.
            </p>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-white text-lg leading-none px-1 ml-3"
              aria-label="Close model health details"
            >
              ✕
            </button>
          </div>
          <ModelHealthView data={data} />
        </div>
      </DrawerShell>
    </div>
  );
}
