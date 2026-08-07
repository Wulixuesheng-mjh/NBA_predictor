/** HistoryPage — prediction history list + accuracy trends dashboard. */

import { useState, useEffect } from "react";
import { fetchPredictions } from "../api/predictions";
import { fetchAccuracyTrends } from "../api/admin";
import type { PredictionHistoryItem, AccuracyTrendResponse } from "../types";
import AccuracyGauge from "../components/charts/AccuracyGauge";
import AccuracyTrendChart from "../components/charts/AccuracyTrendChart";

export default function HistoryPage() {
  const [predictions, setPredictions] = useState<PredictionHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [modelFilter, setModelFilter] = useState("");
  const [trends, setTrends] = useState<AccuracyTrendResponse | null>(null);

  const limit = 20;

  useEffect(() => {
    setLoading(true);
    fetchPredictions({
      model_name: modelFilter || undefined,
      offset,
      limit,
    })
      .then((data) => {
        setPredictions(data.predictions);
        setTotal(data.total);
      })
      .catch(() => [])
      .finally(() => setLoading(false));
  }, [offset, modelFilter]);

  // Fetch accuracy trends on mount
  useEffect(() => {
    fetchAccuracyTrends(30, 7)
      .then(setTrends)
      .catch(() => setTrends(null));
  }, []);

  // Compute summary stats from evaluated predictions
  const evaluated = predictions.filter((p) => p.has_result);
  const totalPages = Math.ceil(total / limit);

  // Use real accuracy data if available, fallback to 0
  const overall = trends?.overall;
  const realWinAcc = overall?.win_accuracy ?? 0;
  const realSpreadMae = overall?.spread_mae ?? 0;
  const realTotalMae = overall?.total_mae ?? 0;

  // Estimate "within-X" rates from MAE (rough heuristic)
  const spreadWithin4 = overall ? Math.min(1, 4 / (realSpreadMae || 999)) : 0;
  const totalWithin8 = overall ? Math.min(1, 8 / (realTotalMae || 999)) : 0;

  const handleExportCSV = () => {
    const url = modelFilter
      ? `/api/v1/predictions/export?model_name=${modelFilter}`
      : "/api/v1/predictions/export";
    window.open(url, "_blank");
  };

  return (
    <div className="space-y-6">
      {/* Accuracy Dashboard */}
      <div className="grid grid-cols-3 gap-4">
        <AccuracyGauge
          winAccuracy={realWinAcc}
          spreadWithin4={spreadWithin4}
          totalWithin8={totalWithin8}
        />
        <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50 flex flex-col justify-center">
          <h4 className="text-sm font-medium text-gray-300 mb-3">预测统计</h4>
          <div className="space-y-3">
            {[
              ["总预测数", total],
              ["已评估", overall?.total_evaluated ?? evaluated.length],
              ["待比赛", total - (overall?.total_evaluated ?? evaluated.length)],
              ["页面显示", predictions.length],
            ].map(([label, value]) => (
              <div key={label} className="flex justify-between">
                <span className="text-sm text-gray-400">{label}</span>
                <span className="text-sm font-mono text-gray-200">{value}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
          <h4 className="text-sm font-medium text-gray-300 mb-3">模型信息</h4>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">主力模型</span>
              <span className="text-gray-200">XGBoost (Optuna-tuned)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">辅助模型</span>
              <span className="text-gray-200">MLP (PyTorch)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">特征维度</span>
              <span className="text-gray-200 font-mono">126</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">校准</span>
              <span className="text-emerald-400">30天滑动窗口</span>
            </div>
            {overall && (
              <div className="flex justify-between">
                <span className="text-gray-400">已评估预测</span>
                <span className="text-gray-200 font-mono">
                  {overall.total_evaluated} 场
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Accuracy Trend Chart */}
      {trends && <AccuracyTrendChart data={trends} />}
      {!trends && (
        <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
          <h4 className="text-sm font-medium text-gray-300 mb-2">准确率趋势</h4>
          <p className="text-gray-500 text-center py-8 text-sm">
            加载趋势数据中...
          </p>
        </div>
      )}

      {/* Filter */}
      <div className="flex gap-3 items-center">
        <select
          value={modelFilter}
          onChange={(e) => {
            setModelFilter(e.target.value);
            setOffset(0);
          }}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
        >
          <option value="">全部模型</option>
          <option value="xgboost">XGBoost</option>
          <option value="mlp">MLP</option>
        </select>
        <button
          onClick={handleExportCSV}
          className="bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-400 text-sm px-4 py-2 rounded-lg transition-colors flex items-center gap-1.5"
        >
          <span>📥</span> 导出 CSV
        </button>
      </div>

      {/* Prediction List */}
      <div className="card">
        <h2 className="card-header">预测记录</h2>
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : predictions.length === 0 ? (
          <p className="text-gray-500 text-center py-8">
            暂无预测记录。前往首页生成预测。
          </p>
        ) : (
          <>
            <div className="space-y-2">
              {predictions.map((p) => (
                <div
                  key={p.id}
                  className="flex items-center gap-4 p-3 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-colors"
                >
                  <div className="w-20 text-xs text-gray-500">
                    {p.predicted_at
                      ? new Date(p.predicted_at).toLocaleDateString("zh-CN")
                      : "—"}
                  </div>
                  <div className="flex-1">
                    <span className="text-gray-200">
                      {p.home_team_name ?? "主队"}
                    </span>
                    <span className="text-gray-600 mx-2">vs</span>
                    <span className="text-gray-200">
                      {p.away_team_name ?? "客队"}
                    </span>
                  </div>
                  <div className="w-20 text-center">
                    <span
                      className={`text-sm font-mono font-bold ${
                        p.win_prob_home > 0.5
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {p.win_prob_home > 0.5 ? "主胜" : "客胜"}{" "}
                      {(p.win_prob_home * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-20 text-center">
                    <span className="text-xs text-gray-500">
                      让分: {p.predicted_spread?.toFixed(1) ?? "—"}
                    </span>
                  </div>
                  <div className="w-20 text-center">
                    <span className="text-xs text-gray-500">
                      总分: {p.predicted_total?.toFixed(1) ?? "—"}
                    </span>
                  </div>
                  <div className="w-16 text-right">
                    <span className="text-xs bg-gray-700 px-2 py-0.5 rounded text-gray-400">
                      {p.model_name}
                    </span>
                  </div>
                  <div className="w-12 text-right">
                    {p.has_result ? (
                      <span className="text-xs text-emerald-400">✓ 已评估</span>
                    ) : (
                      <span className="text-xs text-gray-600">待比赛</span>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex justify-center gap-2 mt-4">
                <button
                  className="btn-secondary text-sm py-1 px-3"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - limit))}
                >
                  上一页
                </button>
                <span className="text-sm text-gray-400 py-1 px-3">
                  {Math.floor(offset / limit) + 1} / {totalPages}
                </span>
                <button
                  className="btn-secondary text-sm py-1 px-3"
                  disabled={offset + limit >= total}
                  onClick={() => setOffset(offset + limit)}
                >
                  下一页
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
