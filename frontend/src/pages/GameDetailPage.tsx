/** GameDetailPage — full game detail with box scores. */

import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { fetchGame } from "../api/games";
import type { GameDetail } from "../types";

export default function GameDetailPage() {
  const { gameId } = useParams<{ gameId: string }>();
  const [game, setGame] = useState<GameDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"home" | "away">("home");

  useEffect(() => {
    if (!gameId) return;
    fetchGame(parseInt(gameId))
      .then(setGame)
      .catch(() => setGame(null))
      .finally(() => setLoading(false));
  }, [gameId]);

  if (loading) {
    return <div className="card"><div className="text-center py-8"><div className="inline-block w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" /></div></div>;
  }

  if (!game) {
    return <div className="card"><p className="text-gray-500 text-center py-8">比赛未找到</p></div>;
  }

  const homePts = game.home_score ?? 0;
  const awayPts = game.away_score ?? 0;

  return (
    <div className="space-y-6">
      {/* Game Header */}
      <div className="card text-center">
        <div className="text-xs text-gray-500 mb-2">
          {new Date(game.game_date).toLocaleDateString("zh-CN", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
        </div>
        <div className="flex items-center justify-center gap-6">
          <div className="text-right flex-1">
            <div className="text-xl font-bold text-white">{game.home_team_name}</div>
            <div className="text-sm text-gray-500">{game.home_team_abbr}</div>
          </div>
          <div className="text-center">
            {game.status === "Final" ? (
              <div className="text-3xl font-bold text-white font-mono">
                {homePts} - {awayPts}
              </div>
            ) : (
              <div className="text-xl text-gray-400">VS</div>
            )}
            <span className={`text-xs px-2 py-0.5 rounded ${
              game.status === "Final" ? "bg-gray-700 text-gray-400" :
              game.status === "InProgress" ? "bg-emerald-500/20 text-emerald-400" :
              "bg-blue-500/20 text-blue-400"
            }`}>
              {game.status === "Final" ? "已结束" : game.status}
            </span>
          </div>
          <div className="text-left flex-1">
            <div className="text-xl font-bold text-white">{game.away_team_name}</div>
            <div className="text-sm text-gray-500">{game.away_team_abbr}</div>
          </div>
        </div>
      </div>

      {/* Team Stats */}
      {game.team_stats.length > 0 && (
        <div className="card">
          <h2 className="card-header">球队数据</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="pb-2 text-left">球队</th>
                  {["PTS","REB","AST","STL","BLK","TOV","FG%","3P%","FT%"].map((h) => (
                    <th key={h} className="pb-2 text-center">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {game.team_stats.map((ts) => (
                  <tr key={ts.team_id} className="border-b border-gray-800/50">
                    <td className="py-2 text-gray-200">{ts.team_name}</td>
                    {[ts.pts, ts.reb, ts.ast, ts.stl, ts.blk, ts.tov,
                      ts.fg_pct ? (ts.fg_pct * 100).toFixed(1) + "%" : "—",
                      ts.fg3_pct ? (ts.fg3_pct * 100).toFixed(1) + "%" : "—",
                      ts.ft_pct ? (ts.ft_pct * 100).toFixed(1) + "%" : "—",
                    ].map((v, i) => (
                      <td key={i} className="py-2 text-center text-gray-400 font-mono text-xs">{v}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Player Stats */}
      {game.player_stats.length > 0 && (
        <div className="card">
          <h2 className="card-header">球员数据</h2>
          <div className="flex gap-2 mb-3">
            {["home", "away"].map((t) => (
              <button
                key={t}
                className={`text-xs px-3 py-1 rounded ${
                  tab === t ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400"
                }`}
                onClick={() => setTab(t as "home" | "away")}
              >
                {t === "home" ? game.home_team_abbr ?? "主队" : game.away_team_abbr ?? "客队"}
              </button>
            ))}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="pb-2 text-left">球员</th>
                  {["MIN","PTS","REB","AST","STL","BLK","TOV","+/-"].map((h) => (
                    <th key={h} className="pb-2 text-center text-xs">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {game.player_stats
                  .filter((ps) => {
                    const isHomePlayer = game.team_stats.find(
                      (ts) => ts.team_id === ps.team_id && ts.is_home
                    );
                    return tab === "home" ? isHomePlayer : !isHomePlayer;
                  })
                  .map((ps) => (
                    <tr key={ps.player_id} className="border-b border-gray-800/50">
                      <td className="py-1.5 text-gray-200 text-xs">{ps.player_name}</td>
                      {[ps.minutes ?? "—", ps.pts, ps.reb, ps.ast, ps.stl, ps.blk, ps.tov, ps.plus_minus ?? "—"].map((v, i) => (
                        <td key={i} className="py-1.5 text-center text-gray-400 font-mono text-xs">{v}</td>
                      ))}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
