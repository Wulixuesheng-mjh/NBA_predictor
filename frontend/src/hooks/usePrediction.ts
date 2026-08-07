/** Hook for generating and managing predictions. */

import { useState, useCallback } from "react";
import type { PredictionRequest, PredictionResponse } from "../types";
import { createPrediction } from "../api/predictions";

export function usePrediction() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = useCallback(async (payload: PredictionRequest) => {
    setLoading(true);
    setError(null);
    try {
      const data = await createPrediction(payload);
      setResult(data);
      return data;
    } catch (e: any) {
      const msg = e.response?.data?.detail ?? e.message ?? "Prediction failed";
      setError(msg);
      setResult(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, predict, clear };
}
