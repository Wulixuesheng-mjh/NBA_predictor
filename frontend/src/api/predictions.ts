/** Prediction API calls. */

import apiClient from "./client";
import type {
  PredictionRequest,
  PredictionResponse,
  PredictionDetail,
  PredictionHistoryListResponse,
} from "../types";

export async function createPrediction(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  const res = await apiClient.post("/predictions/", payload);
  return res.data;
}

export async function fetchPredictions(params: {
  game_id?: number;
  model_name?: string;
  offset?: number;
  limit?: number;
}): Promise<PredictionHistoryListResponse> {
  const res = await apiClient.get("/predictions/", { params });
  return res.data;
}

export async function fetchPrediction(predictionId: number): Promise<PredictionDetail> {
  const res = await apiClient.get(`/predictions/${predictionId}`);
  return res.data;
}
