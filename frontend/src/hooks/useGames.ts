/** Hook for fetching games. */

import { useState, useEffect } from "react";
import type { Game } from "../types";
import { fetchGames, fetchUpcomingGames } from "../api/games";
import type { UpcomingGame } from "../types";

export function useGames(params: {
  date?: string;
  status?: string;
  team_id?: number;
  season?: string;
  offset?: number;
  limit?: number;
}) {
  const [games, setGames] = useState<Game[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Stringify params for stable dependency
  const paramsKey = JSON.stringify(params);

  useEffect(() => {
    setLoading(true);
    fetchGames(params)
      .then((data) => {
        setGames(data.games);
        setTotal(data.total);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [paramsKey]);

  return { games, total, loading, error };
}

export function useUpcomingGames(days: number = 7) {
  const [games, setGames] = useState<UpcomingGame[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUpcomingGames(days)
      .then(setGames)
      .catch(() => [])
      .finally(() => setLoading(false));
  }, [days]);

  return { games, loading };
}
