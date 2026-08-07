/** Team API calls. */

import apiClient from "./client";
import type { Team, TeamDetail, TeamRoster } from "../types";

export async function fetchTeams(conference?: string): Promise<Team[]> {
  const params = conference ? { conference } : {};
  const res = await apiClient.get("/teams/", { params });
  return res.data.teams ?? res.data.data ?? [];
}

export async function fetchTeam(teamId: number): Promise<TeamDetail> {
  const res = await apiClient.get(`/teams/${teamId}`);
  return res.data;
}

export async function fetchTeamRoster(teamId: number): Promise<TeamRoster> {
  const res = await apiClient.get(`/teams/${teamId}/players`);
  return res.data;
}
