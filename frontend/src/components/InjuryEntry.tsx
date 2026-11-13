/** InjuryEntry — manual, per-matchup injury entry for the two selected teams.

 * Replaces the old nba_api auto-pull + AdminPage review workflow. The user picks a
 * player from either selected team's roster and records an injury before predicting.
 * Manual injuries (source='manual') are highest-priority in the prediction features.
 */

import { useState, useEffect, useCallback } from "react";
import { fetchTeamRoster } from "../api/teams";
import { fetchInjuries, createInjury, deleteInjury, confirmCompleteReport } from "../api/injuries";
import type { PlayerBrief, PlayerInjury } from "../types";

const STATUS_OPTIONS = ["Out", "Doubtful", "Questionable", "Probable", "Available"];

interface Props {
  homeTeamId: number | null;
  awayTeamId: number | null;
  homeName: string;
  awayName: string;
  gameId?: number;
}

function TeamInjuryForm({ teamId, teamName, gameId }: { teamId: number; teamName: string; gameId?: number }) {
  const [roster, setRoster] = useState<PlayerBrief[]>([]);
  const [injuries, setInjuries] = useState<PlayerInjury[]>([]);
  const [playerId, setPlayerId] = useState<number>(0);
  const [status, setStatus] = useState("Out");
  const [injuryType, setInjuryType] = useState("");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [completeSaved, setCompleteSaved] = useState(false);

  const refresh = useCallback(async () => {
    setError(null);
    const [r, i] = await Promise.all([
      fetchTeamRoster(teamId).catch(() => null),
      fetchInjuries({ team_id: teamId, game_id: gameId, is_active: true }).catch(() => ({
        injuries: [] as PlayerInjury[],
        total: 0,
        pending_count: 0,
      })),
    ]);
    setRoster(r?.players ?? []);
    setInjuries(i.injuries.filter(injury => gameId ? injury.game_id === gameId : !injury.game_id));
  }, [teamId, gameId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleAdd = async () => {
    if (!playerId || !gameId) return;
    setBusy(true);
    setError(null);
    try {
      await createInjury({
        player_id: playerId,
        game_id: gameId,
        team_id: teamId,
        injury_status: status,
        injury_type: injuryType.trim() || null,
        description: description.trim() || null,
      });
      setPlayerId(0);
      setCompleteSaved(false);
      setInjuryType("");
      setDescription("");
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? "添加失败");
    } finally {
      setBusy(false);
    }
  };

  const handleRemove = async (injuryId: number) => {
    try {
      await deleteInjury(injuryId);
      setCompleteSaved(false);
      await refresh();
    } catch (e: any) {
      setError(e?.message ?? "移除失败");
    }
  };

  const injuredIds = new Set(injuries.map((i) => i.player_id));
  const selectable = roster.filter((p) => !injuredIds.has(p.id));

  return (
    <div className="bg-gray-900/40 rounded-lg border border-gray-700/50 p-3">
      <h5 className="text-sm font-medium text-gray-300 mb-3">{teamName}</h5>

      {gameId && <div className="mb-3">
        <button className="btn-secondary text-xs" disabled={busy || completeSaved} onClick={async () => {
          setBusy(true);
          try {
            const complete = await confirmCompleteReport(gameId, teamId);
            setCompleteSaved(complete);
            if (!complete) setError('名单仍有未识别球员或未知状态，暂不能确认为完整报告。');
          }
          catch (e: any) { setError(e?.message ?? '保存失败'); }
          finally { setBusy(false); }
        }}>{completeSaved ? '本场完整报告已保存' : '确认本场伤病名单完整（含无人伤病）'}</button>
      </div>}
      {/* Add form */}
      <div className="space-y-2">
        <select
          value={playerId}
          onChange={(e) => setPlayerId(Number(e.target.value))}
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-purple-500"
        >
          <option value={0}>选择球员...</option>
          {selectable.map((p) => (
            <option key={p.id} value={p.id}>
              {p.full_name ?? `#${p.id}`}
              {p.position ? ` (${p.position})` : ""}
            </option>
          ))}
        </select>

        <div className="flex gap-2">
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-purple-500"
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <input
            type="text"
            value={injuryType}
            onChange={(e) => setInjuryType(e.target.value)}
            placeholder="伤病类型（如 左脚踝扭伤）"
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-purple-500"
          />
        </div>

        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="备注（可选，如 预计缺席2-3周）"
          className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-purple-500"
        />

        <button
          className="w-full bg-purple-600 hover:bg-purple-700 text-white text-sm px-3 py-1.5 rounded transition-colors disabled:opacity-50"
          onClick={handleAdd}
          disabled={busy || playerId === 0}
        >
          {busy ? "添加中..." : "+ 添加伤病"}
        </button>
      </div>

      {error && <p className="text-xs text-red-400 mt-2">{error}</p>}

      {/* Current active injuries */}
      <div className="mt-3 space-y-1.5">
        {injuries.length === 0 ? (
          <p className="text-xs text-gray-600">当前无伤病记录</p>
        ) : (
          injuries.map((inj) => (
            <div
              key={inj.id}
              className="flex items-center gap-2 text-xs bg-gray-800/60 rounded px-2.5 py-1.5"
            >
              <span className="text-gray-200">{inj.player_name ?? `#${inj.player_id}`}</span>
              <span
                className={`px-1.5 py-0.5 rounded ${
                  inj.injury_status === "Out"
                    ? "bg-red-500/20 text-red-400"
                    : inj.injury_status === "Doubtful"
                    ? "bg-amber-500/20 text-amber-400"
                    : "bg-blue-500/20 text-blue-400"
                }`}
              >
                {inj.injury_status}
              </span>
              {inj.injury_type && <span className="text-gray-500">{inj.injury_type}</span>}
              <button
                className="ml-auto text-red-400 hover:text-red-300 transition-colors"
                onClick={() => handleRemove(inj.id)}
                title="移除伤病"
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default function InjuryEntry({ homeTeamId, awayTeamId, homeName, awayName, gameId }: Props) {
  if (!homeTeamId || !awayTeamId || !gameId) {
    return (
      <div className="card">
        <h2 className="card-header">伤病录入</h2>
        <p className="text-xs text-gray-400">请先选择主客队和具体赛程，再录入本场伤病；假设预览不会保存赛前记录。</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="card-header">
        伤病录入
        <span className="text-xs text-gray-500 font-normal ml-2">{gameId ? '已绑定本场比赛 · 修改将保存新快照' : '未绑定比赛 · 用于假设预览'}</span>
      </h2>
      {gameId && <p className="text-xs text-gray-500 mb-3">下方管理本场手动调整；预测还会读取可用的官方报告。使用 Available 标记确认复出，修改后需重新确认名单。</p>}
      <div className="grid grid-cols-2 gap-4">
        <TeamInjuryForm key={`${homeTeamId}-${gameId}`} teamId={homeTeamId} teamName={homeName} gameId={gameId} />
        <TeamInjuryForm key={`${awayTeamId}-${gameId}`} teamId={awayTeamId} teamName={awayName} gameId={gameId} />
      </div>
    </div>
  );
}
