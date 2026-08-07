/** Injury report panel — shows injuries for both teams in a matchup. */

import { useNavigate } from "react-router-dom";
import type { PlayerInjury } from "../types";

interface Props {
  homeTeamName: string;
  awayTeamName: string;
  homeInjuries: PlayerInjury[];
  awayInjuries: PlayerInjury[];
  pendingCount?: number;
}

function InjuryTeamSection({
  teamName,
  injuries,
}: {
  teamName: string;
  injuries: PlayerInjury[];
}) {
  if (injuries.length === 0) {
    return (
      <div>
        <h5 className="text-sm font-medium text-gray-300 mb-2">{teamName}</h5>
        <p className="text-xs text-gray-600">无伤病</p>
      </div>
    );
  }

  return (
    <div>
      <h5 className="text-sm font-medium text-gray-300 mb-2">{teamName}</h5>
      <div className="space-y-1.5">
        {injuries.map((inj) => (
          <div
            key={inj.id}
            className="flex items-center gap-2 text-sm bg-gray-800/50 rounded px-3 py-1.5"
          >
            <span className="text-gray-200">{inj.player_name || `#${inj.player_id}`}</span>
            <span
              className={`text-xs px-1.5 py-0.5 rounded ${
                inj.injury_status === "Out"
                  ? "bg-red-500/20 text-red-400"
                  : inj.injury_status === "Doubtful"
                  ? "bg-amber-500/20 text-amber-400"
                  : "bg-blue-500/20 text-blue-400"
              }`}
            >
              {inj.injury_status}
            </span>
            {inj.injury_type && (
              <span className="text-xs text-gray-500">{inj.injury_type}</span>
            )}
            <span
              className={`text-xs ml-auto ${
                inj.confirmation === "pending"
                  ? "text-amber-400"
                  : inj.confirmation === "confirmed"
                  ? "text-emerald-400"
                  : "text-blue-400"
              }`}
            >
              {inj.confirmation === "pending" ? "⏳待确认" : "✓"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function InjuryPanel({
  homeTeamName,
  awayTeamName,
  homeInjuries,
  awayInjuries,
  pendingCount,
}: Props) {
  const navigate = useNavigate();

  return (
    <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-gray-300">伤病报告</h4>
        {pendingCount !== undefined && pendingCount > 0 && (
          <button
            className="text-xs text-amber-400 hover:text-amber-300 transition-colors"
            onClick={() => navigate("/admin")}
          >
            ⚠️ {pendingCount} 条待审核 →
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <InjuryTeamSection teamName={homeTeamName} injuries={homeInjuries} />
        <InjuryTeamSection teamName={awayTeamName} injuries={awayInjuries} />
      </div>
    </div>
  );
}
