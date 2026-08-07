/** Player API calls. */

import apiClient from "./client";
import type { Player, PlayerDetail } from "../types";

export async function fetchPlayers(
  teamId?: number,
  search?: string,
  position?: string,
): Promise<{ players: Player[]; total: number }> {
  const params: Record<string, string | number> = {};
  if (teamId) params.team_id = teamId;
  if (search) params.search = search;
  if (position) params.position = position;
  const res = await apiClient.get("/players/", { params });
  return res.data;
}

export async function fetchPlayer(playerId: number): Promise<PlayerDetail> {
  const res = await apiClient.get(`/players/${playerId}`);
  return res.data;
}
