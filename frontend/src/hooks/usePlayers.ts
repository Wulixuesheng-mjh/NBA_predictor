/** Hook for fetching players. */

import { useState, useEffect } from "react";
import type { Player } from "../types";
import { fetchPlayers } from "../api/players";

export function usePlayers(teamId?: number, search?: string) {
  const [players, setPlayers] = useState<Player[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchPlayers(teamId, search)
      .then((data) => {
        setPlayers(data.players);
        setTotal(data.total);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [teamId, search]);

  return { players, total, loading, error };
}
