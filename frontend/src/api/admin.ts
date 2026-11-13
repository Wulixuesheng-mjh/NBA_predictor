/** Admin API calls. */

import apiClient from "./client";
import type { SystemStats, CollectionLogItem, AccuracyTrendResponse } from "../types";

export async function triggerCollection(source: string = "nba_api"): Promise<{
  status: string;
  source: string;
  message?: string;
}> {
  const res = await apiClient.post("/admin/collect", null, { params: { source } });
  return res.data;
}

export async function fetchCollectionStatus(): Promise<{
  last_runs: CollectionLogItem[];
  scheduler_enabled: boolean;
  collection_interval_minutes: number;
}> {
  const res = await apiClient.get("/admin/collect/status");
  return res.data;
}

export async function fetchSystemStats(): Promise<SystemStats> {
  const res = await apiClient.get("/admin/stats");
  return res.data;
}

export async function syncRosters(): Promise<{
  status: string;
  players_synced: number;
}> {
  const res = await apiClient.post("/admin/sync-rosters");
  return res.data;
}

export async function triggerModelTraining(params?: {
  seasons?: string;
  force?: boolean;
}): Promise<{ status: string; version?: string; message?: string }> {
  const res = await apiClient.post("/admin/train-model", null, { params });
  return res.data;
}

export async function fetchAccuracyTrends(
  days: number = 30,
  window: number = 7,
): Promise<AccuracyTrendResponse> {
  const res = await apiClient.get("/predictions/accuracy-trends", {
    params: { days, window },
  });
  return res.data;
}
