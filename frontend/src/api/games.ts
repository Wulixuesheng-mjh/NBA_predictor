/** Game API calls. */

import apiClient from "./client";
import type { GameDetail, UpcomingGame, GameListResponse } from "../types";

export async function fetchGames(params: {
  date?: string;
  status?: string;
  team_id?: number;
  season?: string;
  offset?: number;
  limit?: number;
}): Promise<GameListResponse> {
  const res = await apiClient.get("/games/", { params });
  return res.data;
}

export async function fetchGame(gameId: number): Promise<GameDetail> {
  const res = await apiClient.get(`/games/${gameId}`);
  return res.data;
}

export async function fetchUpcomingGames(days: number = 7): Promise<UpcomingGame[]> {
  const res = await apiClient.get("/games/upcoming", { params: { days } });
  return res.data;
}
