/** HomePage — team selection, prediction generation, and results display. */

import { useState } from "react";
import TeamSelector from "../components/common/TeamSelector";
import LookbackSlider from "../components/common/LookbackSlider";
import { PredictionCard, ConsensusBadge, ExplanationSection } from "../components/prediction";
import InjuryPanel from "../components/InjuryPanel";
import TeamComparisonChart from "../components/charts/TeamComparisonChart";
import RecentForm from "../components/teams/RecentForm";
import { usePrediction } from "../hooks/usePrediction";
import { useTeams } from "../hooks/useTeams";
import { fetchTeam } from "../api/teams";
import { fetchInjuries } from "../api/injuries";
import { fetchGames } from "../api/games";
import type { TeamDetail, PlayerInjury, Game } from "../types";

export default function HomePage() {
  const [homeTeamId, setHomeTeamId] = useState<number | null>(null);
  const [awayTeamId, setAwayTeamId] = useState<number | null>(null);
  const [lookback, setLookback] = useState(10);

  const { result, loading, error, predict } = usePrediction();
  const { teams } = useTeams();

  // Extra data loaded after prediction
  const [homeTeam, setHomeTeam] = useState<TeamDetail | null>(null);
  const [awayTeam, setAwayTeam] = useState<TeamDetail | null>(null);
  const [homeInjuries, setHomeInjuries] = useState<PlayerInjury[]>([]);
  const [awayInjuries, setAwayInjuries] = useState<PlayerInjury[]>([]);
  const [homeGames, setHomeGames] = useState<Game[]>([]);
  const [awayGames, setAwayGames] = useState<Game[]>([]);
  const [extraLoading, setExtraLoading] = useState(false);

  const handlePredict = async () => {
    if (!homeTeamId || !awayTeamId) return;

    const predictionResult = await predict({
      home_team_id: homeTeamId,
      away_team_id: awayTeamId,
      lookback_games: lookback,
    });

    if (!predictionResult) return;

    // Load supporting data
    setExtraLoading(true);
    const [ht, at, hi, ai, hg, ag] = await Promise.all([
      fetchTeam(homeTeamId).catch(() => null),
      fetchTeam(awayTeamId).catch(() => null),
      fetchInjuries({ team_id: homeTeamId, is_active: true }).catch(() => ({ injuries: [], total: 0, pending_count: 0 })),
      fetchInjuries({ team_id: awayTeamId, is_active: true }).catch(() => ({ injuries: [], total: 0, pending_count: 0 })),
      fetchGames({ team_id: homeTeamId, status: "Final", limit: 10 }).catch(() => ({ games: [], total: 0, offset: 0, limit: 10 })),
      fetchGames({ team_id: awayTeamId, status: "Final", limit: 10 }).catch(() => ({ games: [], total: 0, offset: 0, limit: 10 })),
    ]);
    setHomeTeam(ht);
    setAwayTeam(at);
    setHomeInjuries(hi.injuries);
    setAwayInjuries(ai.injuries);
    setHomeGames(hg.games);
    setAwayGames(ag.games);
    setExtraLoading(false);
  };

  const handleRefreshData = async () => {
    try {
      await fetch(`${import.meta.env.VITE_API_BASE ?? ""}/api/v1/admin/collect`, { method: "POST" });
    } catch {
      // Silently fail — user can refresh page
    }
  };

  const homeName = result?.home_team_name ?? teams.find((t) => t.id === homeTeamId)?.full_name ?? "主队";
  const awayName = result?.away_team_name ?? teams.find((t) => t.id === awayTeamId)?.full_name ?? "客队";

  return (
    <div className="space-y-6">
      {/* Team Selection */}
      <div className="card">
        <h2 className="card-header">选择球队</h2>
        <div className="flex items-end gap-6">
          <div className="flex-1">
            <label className="block text-sm text-gray-400 mb-2">主队</label>
            <TeamSelector teams={teams} value={homeTeamId} onChange={setHomeTeamId} />
          </div>
          <div className="text-2xl font-bold text-gray-500 pb-3">VS</div>
          <div className="flex-1">
            <label className="block text-sm text-gray-400 mb-2">客队</label>
            <TeamSelector teams={teams} value={awayTeamId} onChange={setAwayTeamId} />
          </div>
        </div>

        <div className="mt-5 flex items-center gap-4 flex-wrap">
          <LookbackSlider value={lookback} onChange={setLookback} />
          <button
            className="btn-primary"
            onClick={handlePredict}
            disabled={!homeTeamId || !awayTeamId || loading}
          >
            {loading ? "预测中..." : "生成预测"}
          </button>
          <button className="btn-secondary" onClick={handleRefreshData}>
            刷新数据
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Prediction Results */}
      {(result || loading || extraLoading) && (
        <div className="space-y-6">
          {/* Prediction Cards */}
          <div className="card">
            <h2 className="card-header">
              预测结果: {homeName} vs {awayName}
              {result && (
                <span className="text-xs text-gray-500 font-normal ml-2">
                  回溯 {result.lookback_games} 场 | 实际可用: 主{result.actual_games_home}/客{result.actual_games_away}场
                </span>
              )}
            </h2>

            {loading || extraLoading ? (
              <div className="text-center py-12">
                <div className="inline-block w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-gray-500 mt-3">模型推理中...</p>
              </div>
            ) : result ? (
              <>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  {result.xgboost && (
                    <PredictionCard
                      modelName="xgboost"
                      modelLabel="XGBoost"
                      prediction={result.xgboost}
                      recommended={result.recommended === "xgboost"}
                    />
                  )}
                  {result.mlp && (
                    <PredictionCard
                      modelName="mlp"
                      modelLabel="MLP"
                      prediction={result.mlp}
                      recommended={result.recommended === "mlp"}
                    />
                  )}
                </div>

                {result.consensus && <ConsensusBadge consensus={result.consensus} />}

                {/* SHAP Explanation */}
                {result.explanation && (
                  <ExplanationSection explanation={result.explanation} />
                )}

                {/* Data quality warnings */}
                {result.data_quality.warnings.length > 0 && (
                  <div className="mt-3 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                    <p className="text-xs text-amber-400 font-medium mb-1">⚠️ 数据质量提醒</p>
                    {result.data_quality.warnings.map((w, i) => (
                      <p key={i} className="text-xs text-amber-400/80">{w}</p>
                    ))}
                  </div>
                )}

                {result.recommended === "unavailable" && (
                  <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
                    ML 模块不可用 — 请先训练模型 (运行 scripts/train_models.py)
                  </div>
                )}
              </>
            ) : null}
          </div>

          {/* Team Comparison + Recent Form */}
          <div className="grid grid-cols-2 gap-6">
            <TeamComparisonChart homeTeam={homeTeam} awayTeam={awayTeam} />
            <div className="space-y-4">
              <RecentForm teamId={homeTeamId ?? 0} games={homeGames} teamName={homeName} />
              <RecentForm teamId={awayTeamId ?? 0} games={awayGames} teamName={awayName} />
            </div>
          </div>

          {/* Injury Report */}
          <InjuryPanel
            homeTeamName={homeName}
            awayTeamName={awayName}
            homeInjuries={homeInjuries}
            awayInjuries={awayInjuries}
            pendingCount={result?.data_quality.warnings.length}
          />
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && (
        <div className="card">
          <h2 className="card-header">预测结果</h2>
          <div className="text-center py-12">
            <div className="text-4xl mb-3">🏀</div>
            <p className="text-gray-500">选择主客队后点击"生成预测"</p>
            <p className="text-xs text-gray-600 mt-1">
              模型将分析两队近{lookback}场比赛、伤病情况、历史对战等数据
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
