/** GamesPage — browse upcoming and past games with filters. */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useGames, useUpcomingGames } from "../hooks/useGames";
import type { GameStatus } from "../types";

const STATUS_OPTIONS: { label: string; value: GameStatus | "" }[] = [
  { label: "全部", value: "" },
  { label: "已结束", value: "Final" },
  { label: "进行中", value: "InProgress" },
  { label: "已排期", value: "Scheduled" },
];

const SEASONS = ["2025-26", "2024-25", "2023-24"];

export default function GamesPage() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<GameStatus | "">("Final");
  const [season, setSeason] = useState("");
  const [date, setDate] = useState("");
  const [tab, setTab] = useState<"all" | "upcoming">("all");

  const { games, total, loading } = useGames({
    status: status || undefined,
    season: season || undefined,
    date: date || undefined,
    limit: 50,
  });

  const { games: upcoming, loading: upcomingLoading } = useUpcomingGames(7);

  const formatDate = (d: string) => {
    const dt = new Date(d);
    return dt.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
  };

  const formatTime = (d: string) => {
    const dt = new Date(d);
    return dt.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="space-y-6">
      {/* Tabs */}
      <div className="flex gap-2">
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab === "all" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
          onClick={() => setTab("all")}
        >
          全部比赛 ({total})
        </button>
        <button
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab === "upcoming" ? "bg-blue-600 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"
          }`}
          onClick={() => setTab("upcoming")}
        >
          未来赛程 ({upcoming.length})
        </button>
      </div>

      {/* Filters (only for all games tab) */}
      {tab === "all" && (
        <div className="flex gap-3 flex-wrap">
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as GameStatus | "")}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
          >
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select
            value={season}
            onChange={(e) => setSeason(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
          >
            <option value="">全部赛季</option>
            {SEASONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
          />
        </div>
      )}

      {/* Games List */}
      <div className="card">
        <h2 className="card-header">{tab === "upcoming" ? "未来赛程" : "比赛列表"}</h2>

        {(loading || upcomingLoading) ? (
          <div className="text-center py-8">
            <div className="inline-block w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <div className="space-y-1">
            {(tab === "upcoming" ? upcoming : games).map((game: any) => (
              <div
                key={game.id}
                className="flex items-center gap-4 p-3 rounded-lg hover:bg-gray-800/50 cursor-pointer transition-colors"
                onClick={() => navigate(`/games/${game.id}`)}
              >
                {/* Date/Time */}
                <div className="w-28 text-sm text-gray-400">
                  <div>{formatDate(game.game_date)}</div>
                  {game.status !== "Final" && (
                    <div className="text-xs">{formatTime(game.game_date)}</div>
                  )}
                </div>

                {/* Teams */}
                <div className="flex-1 grid grid-cols-3 items-center gap-2">
                  <div className="text-right">
                    <span className="text-gray-200 font-medium">
                      {game.home_team_name ?? game.home_team_abbr ?? `Team #${game.home_team_id}`}
                    </span>
                    <span className="text-xs text-gray-500 ml-1">
                      {game.home_team_abbr ?? ""}
                    </span>
                  </div>
                  <div className="text-center">
                    {game.status === "Final" ? (
                      <span className="text-sm font-mono font-bold text-gray-200">
                        {game.home_score} - {game.away_score}
                      </span>
                    ) : (
                      <span className="text-xs text-gray-600">VS</span>
                    )}
                  </div>
                  <div className="text-left">
                    <span className="text-gray-200 font-medium">
                      {game.away_team_name ?? game.away_team_abbr ?? `Team #${game.away_team_id}`}
                    </span>
                    <span className="text-xs text-gray-500 ml-1">
                      {game.away_team_abbr ?? ""}
                    </span>
                  </div>
                </div>

                {/* Status badge */}
                <div className="w-20 text-right">
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      game.status === "Final"
                        ? "bg-gray-700 text-gray-400"
                        : game.status === "InProgress"
                        ? "bg-emerald-500/20 text-emerald-400 animate-pulse"
                        : game.status === "Scheduled"
                        ? "bg-blue-500/20 text-blue-400"
                        : "bg-gray-700 text-gray-500"
                    }`}
                  >
                    {game.status === "Final" ? "已结束" : game.status === "InProgress" ? "进行中" : game.status === "Scheduled" ? "待比赛" : game.status}
                  </span>
                </div>
              </div>
            ))}

            {((tab === "all" && games.length === 0) || (tab === "upcoming" && upcoming.length === 0)) && (
              <p className="text-gray-500 text-center py-8">暂无比赛数据</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
