/** Injury API calls. */

import apiClient from "./client";
import type {
  PlayerInjury,
  InjuryCreatePayload,
  InjuryUpdatePayload,
  InjuryListResponse,
} from "../types";

export async function fetchInjuries(params: {
  team_id?: number;
  game_id?: number;
  confirmation?: string;
  is_active?: boolean;
  offset?: number;
  limit?: number;
}): Promise<InjuryListResponse> {
  const res = await apiClient.get("/injuries/", { params });
  return res.data;
}

export async function fetchPendingInjuries(): Promise<InjuryListResponse> {
  return fetchInjuries({ confirmation: "pending", is_active: true });
}

export async function createInjury(payload: InjuryCreatePayload): Promise<PlayerInjury> {
  const res = await apiClient.post("/injuries/", payload);
  return res.data;
}

export async function confirmCompleteReport(gameId: number, teamId: number): Promise<boolean> {
  const response = await apiClient.post('/injuries/reports', { game_id: gameId, team_id: teamId, is_complete: true });
  return response.data.is_complete === true;
}

export async function updateInjury(
  injuryId: number,
  payload: InjuryUpdatePayload,
): Promise<PlayerInjury> {
  const res = await apiClient.put(`/injuries/${injuryId}`, payload);
  return res.data;
}

export async function deleteInjury(injuryId: number): Promise<void> {
  await apiClient.delete(`/injuries/${injuryId}`);
}

export async function confirmAllInjuries(): Promise<{ status: string; confirmed_count: number }> {
  const res = await apiClient.post("/injuries/confirm-all");
  return res.data;
}
