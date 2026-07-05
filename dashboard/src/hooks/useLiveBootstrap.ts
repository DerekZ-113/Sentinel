import { useCallback, useEffect, useState } from "react";
import { fetchHealth, fetchStats } from "../services/api";
import type { HealthResponse, StatsResponse } from "../services/api";

export interface LiveBootstrap {
  health: HealthResponse | null;
  stats: StatsResponse | null;
  error: string | null;
  retry: () => void;
}

/**
 * Fetches the app-level bootstrap data (health + stats).
 * On failure the previous error is kept visible until a retry resolves.
 */
export function useLiveBootstrap(): LiveBootstrap {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    Promise.all([fetchHealth(), fetchStats()])
      .then(([h, s]) => {
        setHealth(h);
        setStats(s);
        setError(null);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { health, stats, error, retry: load };
}
