/** TeamsPage — browse all NBA teams with search and conference filter. */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTeams } from "../hooks/useTeams";

export default function TeamsPage() {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [conference, setConference] = useState("");
  const { teams, loading } = useTeams(conference || undefined);

  const filtered = teams.filter(
    (t) =>
      t.full_name.toLowerCase().includes(search.toLowerCase()) ||
      t.abbreviation.toLowerCase().includes(search.toLowerCase())
  );

  const eastTeams = filtered.filter((t) => t.conference === "East");
  const westTeams = filtered.filter((t) => t.conference === "West");
  const otherTeams = filtered.filter((t) => !t.conference);

  return (
    <div className="space-y-6">
      {/* Search & Filter */}
      <div className="flex gap-3 flex-wrap">
        <input
          type="text"
          placeholder="搜索球队..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm text-gray-200 w-64
                     focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <select
          value={conference}
          onChange={(e) => setConference(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200"
        >
          <option value="">全部联盟</option>
          <option value="East">东部</option>
          <option value="West">西部</option>
        </select>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          {/* Eastern Conference */}
          {eastTeams.length > 0 && (
            <div className="card">
              <h2 className="card-header">东部联盟</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                {eastTeams.map((team) => (
                  <TeamCard key={team.id} team={team} onClick={() => navigate(`/teams/${team.id}`)} />
                ))}
              </div>
            </div>
          )}

          {/* Western Conference */}
          {westTeams.length > 0 && (
            <div className="card">
              <h2 className="card-header">西部联盟</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                {westTeams.map((team) => (
                  <TeamCard key={team.id} team={team} onClick={() => navigate(`/teams/${team.id}`)} />
                ))}
              </div>
            </div>
          )}

          {/* Other */}
          {otherTeams.length > 0 && (
            <div className="card">
              <h2 className="card-header">其他</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                {otherTeams.map((team) => (
                  <TeamCard key={team.id} team={team} onClick={() => navigate(`/teams/${team.id}`)} />
                ))}
              </div>
            </div>
          )}

          {filtered.length === 0 && (
            <p className="text-gray-500 text-center py-12">未找到匹配的球队</p>
          )}
        </>
      )}
    </div>
  );
}

function TeamCard({ team, onClick }: { team: any; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="bg-gray-800/60 rounded-xl p-4 text-center hover:bg-gray-800 border border-gray-700/50
                 hover:border-blue-500/50 transition-all cursor-pointer"
    >
      <div className="text-2xl mb-1">🏀</div>
      <div className="text-sm font-medium text-gray-200">{team.full_name}</div>
      <div className="text-xs text-gray-500 mt-0.5">{team.abbreviation}</div>
      {team.division && <div className="text-xs text-gray-600 mt-0.5">{team.division}</div>}
    </button>
  );
}
