/** Hook for fetching injuries. */

import { useState, useEffect, useCallback } from "react";
import type { PlayerInjury } from "../types";
import { fetchInjuries } from "../api/injuries";

export function useInjuries(params: {
  team_id?: number;
  confirmation?: string;
  is_active?: boolean;
} = {}) {
  const [injuries, setInjuries] = useState<PlayerInjury[]>([]);
  const [total, setTotal] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const paramsKey = JSON.stringify(params);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchInjuries(params)
      .then((data) => {
        setInjuries(data.injuries);
        setTotal(data.total);
        setPendingCount(data.pending_count);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [paramsKey]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { injuries, total, pendingCount, loading, error, refresh };
}

export function usePendingInjuries() {
  return useInjuries({ confirmation: "pending", is_active: true });
}
