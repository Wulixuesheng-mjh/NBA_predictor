/** PlayerDetailPage — player info with season stats. */

import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { fetchPlayer } from "../api/players";
import type { PlayerDetail } from "../types";

export default function PlayerDetailPage() {
  const { playerId } = useParams<{ playerId: string }>();
  const [player, setPlayer] = useState<PlayerDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!playerId) return;
    fetchPlayer(parseInt(playerId))
      .then(setPlayer)
      .catch(() => setPlayer(null))
      .finally(() => setLoading(false));
  }, [playerId]);

  if (loading) {
    return <div className="card"><div className="text-center py-8"><div className="inline-block w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" /></div></div>;
  }

  if (!player) {
    return <div className="card"><p className="text-gray-500 text-center py-8">球员未找到</p></div>;
  }

  const s = player.season_stats;

  return (
    <div className="space-y-6">
      {/* Player Header */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">{player.full_name}</h1>
            <div className="flex items-center gap-3 mt-1">
              {player.team_abbreviation && (
                <span className="text-gray-400 text-sm">{player.team_abbreviation}</span>
              )}
              {player.position && (
                <span className="text-xs bg-gray-800 px-2 py-0.5 rounded text-gray-400">
                  {player.position}
                </span>
              )}
              {player.jersey_number && (
                <span className="text-xs text-gray-500">#{player.jersey_number}</span>
              )}
            </div>
          </div>
          {s && (
            <div className="text-right">
              <div className="text-sm text-gray-500">赛季出场</div>
              <div className="text-xl font-bold text-white">{s.games_played} 场</div>
            </div>
          )}
        </div>
      </div>

      {/* Season Stats */}
      {s && (
        <div className="card">
          <h2 className="card-header">赛季场均数据</h2>
          <div className="grid grid-cols-5 gap-3">
            {[
              ["得分", s.avg_pts.toFixed(1)],
              ["篮板", s.avg_reb.toFixed(1)],
              ["助攻", s.avg_ast.toFixed(1)],
              ["抢断", s.avg_stl.toFixed(1)],
              ["盖帽", s.avg_blk.toFixed(1)],
              ["失误", s.avg_tov.toFixed(1)],
              ["投篮%", (s.fg_pct * 100).toFixed(1) + "%"],
              ["三分%", (s.fg3_pct * 100).toFixed(1) + "%"],
              ["罚球%", (s.ft_pct * 100).toFixed(1) + "%"],
              ["场均分钟", s.avg_minutes.toFixed(1)],
            ].map(([label, value]) => (
              <div key={label} className="bg-gray-800/50 rounded-lg p-3 text-center">
                <div className="stat-value text-lg">{value}</div>
                <div className="stat-label">{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Active Injuries */}
      {player.active_injuries.length > 0 && (
        <div className="card">
          <h2 className="card-header">当前伤病</h2>
          {player.active_injuries.map((inj) => (
            <div key={inj.id} className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg">
              <span className={`badge-${inj.injury_status === "Out" ? "loss" : "pending"}`}>
                {inj.injury_status}
              </span>
              <span className="text-gray-300">{inj.injury_type ?? "未知"}</span>
              {inj.description && <span className="text-gray-500 text-sm">— {inj.description}</span>}
              <span className="text-xs text-gray-600 ml-auto">
                {inj.confirmation === "pending" ? "⏳ 待确认" : "✓ 已确认"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
