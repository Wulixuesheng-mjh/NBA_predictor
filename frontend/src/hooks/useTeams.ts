/** Hook for fetching teams. */

import { useState, useEffect } from "react";
import type { Team } from "../types";
import { fetchTeams } from "../api/teams";

export function useTeams(conference?: string) {
  const [teams, setTeams] = useState<Team[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchTeams(conference)
      .then(setTeams)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [conference]);

  return { teams, loading, error };
}
