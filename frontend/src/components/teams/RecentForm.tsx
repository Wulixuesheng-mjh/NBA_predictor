/** Recent form display — shows last N game results as W/L sequence. */

import type { Game } from "../../types";

interface Props {
  teamId: number;
  games: Game[];
  teamName: string;
  limit?: number;
}

export default function RecentForm({ teamId, games, teamName, limit = 10 }: Props) {
  const recent = games.slice(0, limit);

  if (recent.length === 0) {
    return (
      <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
        <h4 className="text-sm font-medium text-gray-300 mb-2">{teamName} 近期战绩</h4>
        <p className="text-xs text-gray-600">暂无数据</p>
      </div>
    );
  }

  let wins = 0;
  const results = recent.map((g) => {
    const isHome = g.home_team_id === teamId;
    const teamScore = isHome ? g.home_score : g.away_score;
    const oppScore = isHome ? g.away_score : g.home_score;
    const isWin = (teamScore ?? 0) > (oppScore ?? 0);
    if (isWin) wins++;
    return { isWin, teamScore, oppScore, date: g.game_date };
  });

  return (
    <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
      <h4 className="text-sm font-medium text-gray-300 mb-2">
        {teamName} 近期战绩{" "}
        <span className={wins >= recent.length / 2 ? "text-emerald-400" : "text-red-400"}>
          ({wins}-{recent.length - wins})
        </span>
      </h4>
      <div className="flex flex-wrap gap-1.5">
        {results.map((r, i) => (
          <div
            key={i}
            className={`w-8 h-8 rounded-md flex flex-col items-center justify-center text-xs font-bold ${
              r.isWin
                ? "bg-emerald-500/20 text-emerald-400"
                : "bg-red-500/20 text-red-400"
            }`}
            title={`${r.teamScore}-${r.oppScore} (${r.date.slice(0, 10)})`}
          >
            {r.isWin ? "W" : "L"}
          </div>
        ))}
      </div>
    </div>
  );
}
