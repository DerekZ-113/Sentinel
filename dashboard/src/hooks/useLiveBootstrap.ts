import { useCallback, useEffect, useState } from "react";
import { fetchHealth, fetchStats } from "../services/api";
import type { HealthResponse, StatsResponse } from "../services/api";

export interface LiveBootstrap {
  health: HealthResponse | null;
  stats: StatsResponse | null;
  /** Initial-load failure: blocks the app with the error screen. */
  error: string | null;
  /** Later refresh failure: last good data stays visible. */
  refreshError: string | null;
  lastUpdatedAt: Date | null;
  retry: () => void;
  refresh: () => void;
}

type LoadMode = "initial" | "refresh";

/**
 * Fetches the app-level bootstrap data (health + stats). Initial failures
 * surface as `error` (kept visible until a retry resolves); refresh failures
 * surface as `refreshError` while preserving the last good data.
 */
export function useLiveBootstrap(): LiveBootstrap {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);

  const load = useCallback((mode: LoadMode) => {
    Promise.all([fetchHealth(), fetchStats()])
      .then(([h, s]) => {
        setHealth(h);
        setStats(s);
        setError(null);
        setRefreshError(null);
        setLastUpdatedAt(new Date());
      })
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : "Unknown error";
        if (mode === "initial") {
          setError(message);
        } else {
          setRefreshError(`Refresh failed: ${message}`);
        }
      });
  }, []);

  useEffect(() => {
    load("initial");
  }, [load]);

  const retry = useCallback(() => load("initial"), [load]);
  const refresh = useCallback(() => load("refresh"), [load]);

  return { health, stats, error, refreshError, lastUpdatedAt, retry, refresh };
}
