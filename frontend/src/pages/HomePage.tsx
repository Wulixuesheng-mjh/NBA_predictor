/** HomePage — team selection, prediction generation, and results display. */

import { useState, useEffect } from "react";
import TeamSelector from "../components/common/TeamSelector";
import LookbackSlider from "../components/common/LookbackSlider";
import { PredictionCard, ConsensusBadge, ExplanationSection, CoverOverSection, MarketOddsSection } from "../components/prediction";
import InjuryEntry from "../components/InjuryEntry";
import InjuryPanel from "../components/InjuryPanel";
import TeamComparisonChart from "../components/charts/TeamComparisonChart";
import RecentForm from "../components/teams/RecentForm";
import { usePrediction } from "../hooks/usePrediction";
import { triggerCollection } from "../api/admin";
import { fetchShadowStatus, type ShadowObservationStatus } from "../api/predictions";
import { useTeams } from "../hooks/useTeams";
import { fetchTeam } from "../api/teams";
import { fetchInjuries } from "../api/injuries";
import { fetchGames } from "../api/games";
import type { TeamDetail, PlayerInjury, Game } from "../types";

export default function HomePage() {
  const [shadowStatus, setShadowStatus] = useState<ShadowObservationStatus | null>(null);
  useEffect(() => {
    let alive = true;
    const refresh = () => fetchShadowStatus().then(data => { if (alive) setShadowStatus(data); }).catch(() => {});
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 60000);
    return () => { alive = false; window.clearInterval(timer); };
  }, []);
  const [homeTeamId, setHomeTeamId] = useState<number | null>(null);
  const [awayTeamId, setAwayTeamId] = useState<number | null>(null);
  const [lookback, setLookback] = useState(10);
  const [spreadLine, setSpreadLine] = useState("");
  const [totalLine, setTotalLine] = useState("");
  const [homeOdds, setHomeOdds] = useState("");
  const [awayOdds, setAwayOdds] = useState("");
  const [upcoming, setUpcoming] = useState<Game[]>([]);
  const [selectedGameId, setSelectedGameId] = useState<number | null>(null);
  const targetGame = upcoming.find(g => g.id === selectedGameId && g.home_team_id === homeTeamId && g.away_team_id === awayTeamId);

  useEffect(() => {
    let cancelled = false;
    setUpcoming([]); setSelectedGameId(null);
    if (homeTeamId && awayTeamId) {
      fetchGames({ team_id: homeTeamId, status: 'Scheduled', limit: 100 }).then(data => {
        if (cancelled) return;
        const matches = data.games.filter(g => g.home_team_id === homeTeamId && g.away_team_id === awayTeamId
          && g.time_precision === 'exact' && g.scheduled_start_utc && Date.parse(g.scheduled_start_utc) > Date.now());
        setUpcoming(matches);
        if (matches.length === 1) setSelectedGameId(matches[0].id);
      }).catch(() => { if (!cancelled) setUpcoming([]); });
    }
    return () => { cancelled = true; };
  }, [homeTeamId, awayTeamId]);

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

    // Both lines must be present (and valid) to enable 盘口 prediction
    const spread = spreadLine.trim() === "" ? undefined : Number(spreadLine);
    const total = totalLine.trim() === "" ? undefined : Number(totalLine);
    const hasLines =
      spread !== undefined && total !== undefined && !Number.isNaN(spread) && !Number.isNaN(total);

    // 胜负赔率：两项都填且都大于 1 才启用市场对比
    const hOdds = homeOdds.trim() === "" ? undefined : Number(homeOdds);
    const aOdds = awayOdds.trim() === "" ? undefined : Number(awayOdds);
    const hasOdds =
      hOdds !== undefined &&
      aOdds !== undefined &&
      !Number.isNaN(hOdds) &&
      !Number.isNaN(aOdds) &&
      hOdds > 1 &&
      aOdds > 1;

    const predictionResult = await predict({
      home_team_id: homeTeamId,
      away_team_id: awayTeamId,
      ...(targetGame ? { game_id: targetGame.id } : {}),
      mode: targetGame ? "pregame" : "hypothetical_preview",
      persist: Boolean(targetGame),
      lookback_games: lookback,
      ...(hasLines ? { market_home_handicap: spread as number, total_line: total as number } : {}),
      ...(hasOdds ? { home_odds: hOdds as number, away_odds: aOdds as number } : {}),
    });

    if (!predictionResult) return;

    // Load supporting data
    setExtraLoading(true);
    const [ht, at, hi, ai, hg, ag] = await Promise.all([
      fetchTeam(homeTeamId).catch(() => null),
      fetchTeam(awayTeamId).catch(() => null),
      fetchInjuries({ team_id: homeTeamId, game_id: targetGame?.id, is_active: true }).catch(() => ({ injuries: [], total: 0, pending_count: 0 })),
      fetchInjuries({ team_id: awayTeamId, game_id: targetGame?.id, is_active: true }).catch(() => ({ injuries: [], total: 0, pending_count: 0 })),
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
      // Goes through the shared axios client so the X-API-Key header is attached.
      await triggerCollection("kaggle");
    } catch {
      // Silently fail — user can refresh page
    }
  };

  const homeName = result?.home_team_name ?? teams.find((t) => t.id === homeTeamId)?.full_name ?? "主队";
  const awayName = result?.away_team_name ?? teams.find((t) => t.id === awayTeamId)?.full_name ?? "客队";
  const totalLineValue =
    totalLine.trim() === "" || Number.isNaN(Number(totalLine)) ? null : Number(totalLine);

  return (
    <div className="space-y-6">
      {shadowStatus?.runs.map(run => (
        <div key={run.name} className="card text-sm text-gray-300">
          <div className="font-medium">自动影子观察 · {run.version}</div>
          <div className="mt-2">已留档 {run.captured} 场 · 已结算 {run.settled} 场 · 待结束 {run.awaiting} 场</div>
          <div className="mt-1 text-xs text-gray-500">
            开赛前约一小时采集，独立于正式200场验收。
            {!run.configuration_matches || run.integrity_errors > 0 ? " 配置或记录校验未通过，已停止新增观察。" : ""}
          </div>
          {run.settled > 0 && run.metrics ? (
            <div className="mt-3 grid grid-cols-3 gap-3">
              {Object.entries(run.metrics).map(([name, metric]) => (
                <div key={name} className="rounded bg-gray-900/40 p-3">
                  <div>{name === "baseline" ? "主模型" : name === "direct" ? "直接候选" : "残差候选"}</div>
                  <div className="mt-1">总分 MAE {metric.mae.toFixed(2)}</div>
                  <div className="text-xs text-gray-400">平均偏差 {metric.bias >= 0 ? "+" : ""}{metric.bias.toFixed(2)}（预测−实际）</div>
                </div>
              ))}
            </div>
          ) : <div className="mt-2 text-xs text-gray-500">尚无已结束比赛可结算，暂不显示准确率。</div>}
        </div>
      ))}
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

        <div className="mt-4">
          <label className="block text-sm text-gray-400 mb-2">预测比赛</label>
          <select className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm" value={targetGame?.id ?? ''}
            onChange={e => setSelectedGameId(e.target.value ? Number(e.target.value) : null)}>
            <option value="">假设对阵预览</option>
            {upcoming.map(g => <option key={g.id} value={g.id}>{new Date(g.scheduled_start_utc!).toLocaleString('zh-CN')} 开球</option>)}
          </select>
          <p className="text-xs text-gray-500 mt-2">{targetGame ? '伤病录入绑定本场；生成预测会保存当前时点的赛前记录。' : '选择具体赛程后可录入本场伤病并保存赛前预测。'}</p>
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

        {/* 盘口录入 (可选) — 用于让分胜负 / 大小分二分类预测 */}
        <div className="mt-4 p-3 bg-gray-800/40 rounded-lg border border-gray-700/50">
          <p className="text-xs text-gray-400 mb-2">
            盘口录入（可选，两项都填才启用让分胜负 / 大小分预测）
          </p>
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400 whitespace-nowrap">让分盘口</label>
              <input
                type="number"
                step="0.5"
                placeholder="如 -6.5"
                value={spreadLine}
                onChange={(e) => setSpreadLine(e.target.value)}
                className="w-28 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-700 text-sm text-gray-200 focus:outline-none focus:border-purple-500"
              />
              <span className="text-xs text-gray-600">体彩原始符号：负数表示主让</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400 whitespace-nowrap">大小分盘口</label>
              <input
                type="number"
                step="0.5"
                placeholder="如 220.5"
                value={totalLine}
                onChange={(e) => setTotalLine(e.target.value)}
                className="w-28 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-700 text-sm text-gray-200 focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>
        </div>

        {/* 胜负赔率录入 (可选) — 与模型概率对比，判断市场是否存在价值 */}
        <div className="mt-3 p-3 bg-gray-800/40 rounded-lg border border-gray-700/50">
          <p className="text-xs text-gray-400 mb-2">
            胜负赔率录入（可选，取自体彩；两项都填才启用市场对比）
          </p>
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400 whitespace-nowrap">主胜赔率</label>
              <input
                type="number"
                step="0.01"
                min="1.01"
                placeholder="如 1.85"
                value={homeOdds}
                onChange={(e) => setHomeOdds(e.target.value)}
                className="w-28 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-700 text-sm text-gray-200 focus:outline-none focus:border-amber-500"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400 whitespace-nowrap">客胜赔率</label>
              <input
                type="number"
                step="0.01"
                min="1.01"
                placeholder="如 2.05"
                value={awayOdds}
                onChange={(e) => setAwayOdds(e.target.value)}
                className="w-28 px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-700 text-sm text-gray-200 focus:outline-none focus:border-amber-500"
              />
            </div>
            <span className="text-xs text-gray-600">小数赔率，体彩官网可查</span>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Manual injury entry (near team selection) */}
      <InjuryEntry
        homeTeamId={homeTeamId}
        awayTeamId={awayTeamId}
        homeName={homeName}
        awayName={awayName}
        gameId={targetGame?.id}
      />

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
              {result?.model_release && (
                <span className="block text-xs text-gray-600 font-normal mt-1">
                  模型 {result.model_release.xgboost_version ?? "—"}
                  {result.model_release.mlp_version
                    ? ` + MLP ${result.model_release.mlp_version}`
                    : "（MLP 不可用）"}
                  {result.model_release.training_data_until
                    ? ` | 训练数据截止 ${result.model_release.training_data_until}`
                    : ""}
                </span>
              )}
              {result?.baseline && (
                <span
                  className={`block text-xs font-normal mt-1 ${
                    result.baseline.status === "divergent" ? "text-amber-400" : "text-gray-600"
                  }`}
                >
                  节奏×效率基线 {result.baseline.baseline_total.toFixed(1)}
                  （模型 {result.baseline.model_total?.toFixed(1) ?? "—"}，差{" "}
                  {result.baseline.diff_vs_model ?? "—"}）
                  {result.baseline.status === "divergent" ? " ⚠️ 与模型分歧较大" : ""}
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
                {result.shadow_totals && (
                  <div className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/5 p-4">
                    <div className="font-medium text-amber-300">新模型总分影子对照</div>
                    <p className="mt-1 text-xs text-gray-400">
                      用于验证，尚未晋升为主模型；当前候选未启用伤病调整。
                    </p>
                    {result.shadow_totals.status === "unavailable" ? (
                      <p className="mt-2 text-sm text-gray-400">{result.shadow_totals.reason}</p>
                    ) : (
                      <div className="mt-3 grid grid-cols-2 gap-3">
                        {result.shadow_totals.candidates.map((candidate) => (
                          <div key={candidate.version} className="rounded bg-gray-900/40 p-3">
                            <div className="text-sm text-gray-300">
                              {candidate.strategy === "direct" ? "直接总分" : "节奏效率基线＋残差"}
                            </div>
                            <div className="mt-1 text-2xl font-semibold text-white">{candidate.predicted_total.toFixed(1)}</div>
                            <div className="mt-1 text-xs text-gray-500">训练截止 {candidate.training_data_until}</div>
                            <div className="mt-1 break-all text-xs text-gray-500">{candidate.version}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  {result.xgboost && (
                    <PredictionCard
                      modelName="xgboost"
                      modelLabel="XGBoost"
                      prediction={result.xgboost}
                      recommended={result.recommended === "xgboost"}
                      confidence={result.confidence}
                      totalLine={totalLineValue}
                    />
                  )}
                  {result.mlp && (
                    <PredictionCard
                      modelName="mlp"
                      modelLabel="MLP"
                      prediction={result.mlp}
                      recommended={result.recommended === "mlp"}
                      confidence={result.confidence}
                      totalLine={totalLineValue}
                    />
                  )}
                </div>

                {result.confidence && result.confidence.level === "not_recommended" && (
                  <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                    <p className="text-sm text-red-400 font-medium">⚠️ 不建议参考此预测</p>
                    <p className="text-xs text-red-400/70 mt-1">
                      置信度评分仅 {result.confidence.score.toFixed(0)}/100。
                      {result.actual_games_home < 5 || result.actual_games_away < 5
                        ? " 可用比赛数据不足（至少需要5场），建议等待更多比赛后再预测。"
                        : " 模型对此预测信心较低，预测结果仅供参考。"}
                    </p>
                  </div>
                )}

                {result.consensus && <ConsensusBadge consensus={result.consensus} />}

                {/* SHAP Explanation */}
                {result.explanation && (
                  <ExplanationSection explanation={result.explanation} />
                )}

                {/* Cover/over (盘口) binary prediction */}
                {result.cover_over && (
                  <CoverOverSection coverOver={result.cover_over} />
                )}

                {/* 胜负赔率 vs 模型概率 */}
                {result.market && (
                  <div className="mt-4">
                    <MarketOddsSection market={result.market} homeName={homeName} awayName={awayName} />
                  </div>
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
