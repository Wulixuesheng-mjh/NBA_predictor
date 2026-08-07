/** TeamDetailPage — roster, season stats, recent games for one team. */

import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { fetchTeam, fetchTeamRoster } from "../api/teams";
import { fetchPlayers } from "../api/players";
import { fetchGames } from "../api/games";
import { fetchInjuries } from "../api/injuries";
import type { TeamDetail, TeamRoster, Player, Game, PlayerInjury } from "../types";

export default function TeamDetailPage() {
  const { teamId } = useParams<{ teamId: string }>();
  const navigate = useNavigate();
  const tid = parseInt(teamId ?? "0");

  const [team, setTeam] = useState<TeamDetail | null>(null);
  const [roster, setRoster] = useState<TeamRoster | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [games, setGames] = useState<Game[]>([]);
  const [injuries, setInjuries] = useState<PlayerInjury[]>([]);
  const [tab, setTab] = useState<"roster" | "games">("roster");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!tid) return;
    setLoading(true);
    Promise.all([
      fetchTeam(tid).catch(() => null),
      fetchTeamRoster(tid).catch(() => null),
      fetchPlayers(tid).catch(() => ({ players: [], total: 0 })),
      fetchGames({ team_id: tid, status: "Final", limit: 20 }).catch(() => ({ games: [], total: 0, offset: 0, limit: 20 })),
      fetchInjuries({ team_id: tid, is_active: true }).catch(() => ({ injuries: [], total: 0, pending_count: 0 })),
    ]).then(([t, r, p, g, i]) => {
      setTeam(t);
      setRoster(r);
      setPlayers(p.players);
      setGames(g.games);
      setInjuries(i.injuries);
      setLoading(false);
    });
  }, [tid]);

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!team) {
    return <div className="card"><p className="text-gray-500 text-center py-8">球队未找到</p></div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">{team.full_name}</h1>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-gray-400">{team.abbreviation}</span>
              {team.conference && (
                <span className="text-xs bg-gray-800 px-2 py-0.5 rounded text-gray-400">
                  {team.conference === "East" ? "东部" : "西部"} | {team.division ?? ""}
                </span>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">近期战绩</div>
            <div className="text-xl font-bold text-gray-200">{team.recent_record ?? "N/A"}</div>
          </div>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-4 gap-4 mt-4">
          {[
            ["场均得分", team.avg_pts],
            ["场均失分", team.avg_pts_allowed],
            ["净胜分", team.net_rating],
            ["球员数", roster?.total],
          ].map(([label, value]) => (
            <div key={label} className="bg-gray-800/50 rounded-lg p-3 text-center">
              <div className="stat-value text-lg">{value ?? "—"}</div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>

        {/* Active injuries */}
        {injuries.length > 0 && (
          <div className="mt-4 p-3 bg-red-500/5 border border-red-500/20 rounded-lg">
            <p className="text-sm text-red-400 font-medium mb-2">当前伤病 ({injuries.length})</p>
            <div className="flex flex-wrap gap-2">
              {injuries.map((inj) => (
                <span key={inj.id} className="text-xs bg-red-500/10 text-red-400 px-2 py-1 rounded">
                  {inj.player_name} — {inj.injury_status}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2">
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab === "roster" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400"
          }`}
          onClick={() => setTab("roster")}
        >
          球员阵容 ({players.length})
        </button>
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab === "games" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400"
          }`}
          onClick={() => setTab("games")}
        >
          近期比赛 ({games.length})
        </button>
      </div>

      {/* Roster tab */}
      {tab === "roster" && (
        <div className="card">
          <h2 className="card-header">球员阵容</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500 border-b border-gray-800">
                  <th className="pb-2 font-medium">#</th>
                  <th className="pb-2 font-medium">球员</th>
                  <th className="pb-2 font-medium">位置</th>
                  <th className="pb-2 font-medium">球衣号</th>
                </tr>
              </thead>
              <tbody>
                {players.map((p) => (
                  <tr
                    key={p.id}
                    className="border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer"
                    onClick={() => navigate(`/players/${p.id}`)}
                  >
                    <td className="py-2 text-gray-600 text-xs">{p.nba_player_id}</td>
                    <td className="py-2 text-gray-200">{p.full_name}</td>
                    <td className="py-2 text-gray-400">{p.position || "—"}</td>
                    <td className="py-2 text-gray-400">{p.jersey_number || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Games tab */}
      {tab === "games" && (
        <div className="card">
          <h2 className="card-header">近期比赛</h2>
          <div className="space-y-1">
            {games.map((g) => {
              const isHome = g.home_team_id === tid;
              const teamScore = isHome ? g.home_score : g.away_score;
              const oppScore = isHome ? g.away_score : g.home_score;
              const isWin = (teamScore ?? 0) > (oppScore ?? 0);
              return (
                <div
                  key={g.id}
                  className="flex items-center gap-3 p-2.5 rounded hover:bg-gray-800/30"
                >
                  <span className={`text-xs font-bold w-6 ${isWin ? "text-emerald-400" : "text-red-400"}`}>
                    {isWin ? "W" : "L"}
                  </span>
                  <span className="text-sm text-gray-400 w-24">
                    {new Date(g.game_date).toLocaleDateString("zh-CN")}
                  </span>
                  <span className="text-sm text-gray-200 font-mono">
                    {teamScore} - {oppScore}
                  </span>
                  <span className="text-xs text-gray-600">
                    {isHome ? "vs" : "@"}
                  </span>
                </div>
              );
            })}
            {games.length === 0 && (
              <p className="text-gray-500 text-center py-4">暂无比赛数据</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
