/** AdminPage — injury review workbench, collection triggers, system stats. */

import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAdminStats } from "../hooks/useAdminStats";
import { usePendingInjuries } from "../hooks/useInjuries";
import {
  confirmAllInjuries,
  updateInjury,
  deleteInjury,
  createInjury,
} from "../api/injuries";
import { fetchPlayers } from "../api/players";
import { triggerCollection, triggerModelTraining } from "../api/admin";
import type { PlayerInjury, Player } from "../types";

type InjuryAction = "confirm" | "delete" | "edit";
type SortField = "player_name" | "injury_status" | "confirmation";

export default function AdminPage() {
  const navigate = useNavigate();
  const { stats, loading: statsLoading, refresh: refreshStats } = useAdminStats();
  const { injuries, loading: injuriesLoading, refresh: refreshInjuries } = usePendingInjuries();

  const [apiKey, setApiKey] = useState(() => localStorage.getItem("admin_api_key") || "");
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainingMsg, setTrainingMsg] = useState("");
  const [sortField, setSortField] = useState<SortField>("player_name");
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editStatus, setEditStatus] = useState("");

  const handleSaveApiKey = () => {
    localStorage.setItem("admin_api_key", apiKey);
    setShowKeyInput(false);
    refreshStats(); // re-fetch with new key
    refreshInjuries();
  };

  const handleClearApiKey = () => {
    localStorage.removeItem("admin_api_key");
    setApiKey("");
    setShowKeyInput(false);
  };

  // ── Manual injury creation form ──
  const [showAddForm, setShowAddForm] = useState(false);
  const [players, setPlayers] = useState<Player[]>([]);
  const [playerSearch, setPlayerSearch] = useState("");
  const [newInjury, setNewInjury] = useState({
    player_id: 0,
    injury_status: "Questionable",
    injury_type: "",
    description: "",
    expected_return: "",
  });
  const [creating, setCreating] = useState(false);

  // Search players when typing
  useEffect(() => {
    if (playerSearch.length < 2) { setPlayers([]); return; }
    const timer = setTimeout(async () => {
      try {
        const result = await fetchPlayers(undefined, playerSearch);
        setPlayers(result.players);
      } catch { setPlayers([]); }
    }, 300);
    return () => clearTimeout(timer);
  }, [playerSearch]);

  const handleConfirmAll = async () => {
    setConfirming(true);
    try {
      await confirmAllInjuries();
      await refreshInjuries();
      await refreshStats();
    } finally {
      setConfirming(false);
    }
  };

  const handleInjuryAction = async (injury: PlayerInjury, action: InjuryAction) => {
    if (action === "confirm") {
      await updateInjury(injury.id, { injury_status: injury.injury_status });
    } else if (action === "delete") {
      await deleteInjury(injury.id);
    }
    await refreshInjuries();
    await refreshStats();
  };

  const handleEdit = async (injuryId: number) => {
    if (!editStatus) return;
    await updateInjury(injuryId, { injury_status: editStatus });
    setEditingId(null);
    setEditStatus("");
    await refreshInjuries();
    await refreshStats();
  };

  const handleCollect = async () => {
    setCollecting(true);
    try {
      const result = await triggerCollection("nba_api");
      await refreshStats();
      alert(`采集${result.status === "triggered" ? "完成" : "失败"}: ${result.message ?? ""}`);
    } finally {
      setCollecting(false);
    }
  };

  const handleTrainModel = async () => {
    setTraining(true);
    setTrainingMsg("模型训练中...");
    try {
      const result = await triggerModelTraining({ force: true });
      setTrainingMsg(result.message ?? "训练完成");
    } catch (e: any) {
      setTrainingMsg(`训练失败: ${e.message}`);
    } finally {
      setTraining(false);
    }
  };

  const handleCreateInjury = async () => {
    if (!newInjury.player_id) {
      alert("请选择球员");
      return;
    }
    setCreating(true);
    try {
      await createInjury({
        player_id: newInjury.player_id,
        injury_status: newInjury.injury_status,
        injury_type: newInjury.injury_type || null,
        description: newInjury.description || null,
        expected_return: newInjury.expected_return || null,
      });
      setShowAddForm(false);
      setNewInjury({ player_id: 0, injury_status: "Questionable", injury_type: "", description: "", expected_return: "" });
      setPlayerSearch("");
      setPlayers([]);
      await refreshInjuries();
      await refreshStats();
    } catch (e: any) {
      alert(`创建失败: ${e.message}`);
    } finally {
      setCreating(false);
    }
  };

  // Confirmation status badge colors
  const confirmationBadge = (status: string) => {
    switch (status) {
      case "confirmed": return "bg-emerald-500/20 text-emerald-400";
      case "modified": return "bg-blue-500/20 text-blue-400";
      case "manual": return "bg-purple-500/20 text-purple-400";
      case "pending": return "bg-amber-500/20 text-amber-400";
      default: return "bg-gray-500/20 text-gray-400";
    }
  };
  const confirmationLabel = (status: string) => {
    switch (status) {
      case "confirmed": return "已确认";
      case "modified": return "已修改";
      case "manual": return "手动录入";
      case "pending": return "待确认";
      default: return status;
    }
  };

  // Sort injuries
  const sortedInjuries = [...injuries].sort((a, b) => {
    if (sortField === "player_name") {
      return (a.player_name ?? "").localeCompare(b.player_name ?? "");
    }
    if (sortField === "injury_status") {
      return (a.injury_status ?? "").localeCompare(b.injury_status ?? "");
    }
    return (a.confirmation ?? "").localeCompare(b.confirmation ?? "");
  });

  if (statsLoading) {
    return <div className="card"><p className="text-gray-500 text-center py-8">加载中...</p></div>;
  }

  return (
    <div className="space-y-6">
      {/* System Stats */}
      <div className="card">
        <h2 className="card-header">系统状态</h2>

        {/* API Key management */}
        <div className="mb-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-300">🔑 管理员密钥</span>
              {apiKey ? (
                <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded">已配置</span>
              ) : (
                <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded">未配置</span>
              )}
            </div>
            <button
              className="text-xs text-gray-400 hover:text-gray-200 transition-colors"
              onClick={() => setShowKeyInput(!showKeyInput)}
            >
              {showKeyInput ? "取消" : apiKey ? "更换密钥" : "设置密钥"}
            </button>
          </div>
          {showKeyInput && (
            <div className="mt-3 flex gap-2">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="输入管理员 API Key..."
                className="flex-1 bg-gray-700 border border-gray-600 rounded px-3 py-1.5 text-sm text-gray-200
                           placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
              <button
                className="bg-blue-600 hover:bg-blue-700 text-white text-xs px-3 py-1.5 rounded transition-colors"
                onClick={handleSaveApiKey}
              >
                保存
              </button>
              {apiKey && (
                <button
                  className="bg-red-600/20 hover:bg-red-600/40 text-red-400 text-xs px-3 py-1.5 rounded transition-colors"
                  onClick={handleClearApiKey}
                >
                  清除
                </button>
              )}
            </div>
          )}
          <p className="text-xs text-gray-600 mt-2">
            设置密钥后所有管理操作需要验证。密钥为空时不启用认证（开发模式）。
          </p>
        </div>
        <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
          {[
            ["球队", stats?.teams_count],
            ["球员", stats?.players_count],
            ["比赛", stats?.games_count],
            ["已完成", stats?.completed_games_count],
            ["活跃伤病", stats?.active_injuries_count],
            ["待审核", stats?.pending_injuries_count],
            ["预测数", stats?.predictions_count],
            ["已评估", stats?.evaluated_predictions_count],
            ["上次采集", stats?.last_collection ? new Date(stats.last_collection).toLocaleString("zh-CN") : "—"],
            ["模型版本", stats?.active_model_version || "未训练"],
          ].map(([label, value]) => (
            <div key={label} className="bg-gray-800/50 rounded-lg p-3 text-center">
              <div className={`text-lg font-bold ${typeof value === 'number' ? 'text-white' : 'text-xs text-gray-400'}`}>
                {value ?? "—"}
              </div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 mt-4 flex-wrap">
          <button className="btn-secondary" onClick={handleCollect} disabled={collecting}>
            {collecting ? "采集中..." : "手动触发采集"}
          </button>
          <button className="btn-primary" onClick={handleConfirmAll} disabled={confirming || injuries.length === 0}>
            {confirming ? "确认中..." : `一键确认全部伤病 (${injuries.length})`}
          </button>
          <button
            className="bg-amber-600 hover:bg-amber-700 text-white font-medium py-2 px-6 rounded-lg
                       transition-colors duration-200 disabled:opacity-50 text-sm"
            onClick={handleTrainModel}
            disabled={training}
          >
            {training ? "训练中..." : "触发模型训练"}
          </button>
        </div>
        {trainingMsg && (
          <p className="mt-2 text-sm text-gray-400">{trainingMsg}</p>
        )}
      </div>

      {/* Injury Review Workbench */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-100">伤病审核工作台</h2>
          <div className="flex gap-2">
            <select
              value={sortField}
              onChange={(e) => setSortField(e.target.value as SortField)}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-400"
            >
              <option value="player_name">按球员</option>
              <option value="injury_status">按状态</option>
              <option value="confirmation">按确认</option>
            </select>
            <button
              className="bg-purple-600 hover:bg-purple-700 text-white text-xs px-3 py-1 rounded transition-colors"
              onClick={() => setShowAddForm(!showAddForm)}
            >
              {showAddForm ? "取消" : "+ 手动新增"}
            </button>
          </div>
        </div>

        {/* Manual add form */}
        {showAddForm && (
          <div className="mb-4 p-4 bg-purple-500/5 border border-purple-500/20 rounded-lg">
            <h3 className="text-sm font-medium text-purple-300 mb-3">手动录入伤病信息</h3>
            <div className="space-y-3">
              {/* Player search */}
              <div className="relative">
                <label className="text-xs text-gray-400 block mb-1">球员搜索</label>
                <input
                  type="text"
                  value={playerSearch}
                  onChange={(e) => setPlayerSearch(e.target.value)}
                  placeholder="输入球员姓名搜索..."
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200"
                  disabled={newInjury.player_id > 0}
                />
                {newInjury.player_id > 0 && (
                  <p className="text-xs text-emerald-400 mt-1">已选择球员 (ID: {newInjury.player_id}) — 重新输入以更改</p>
                )}
                {players.length > 0 && newInjury.player_id === 0 && (
                  <div className="absolute z-10 w-full mt-1 bg-gray-800 border border-gray-700 rounded max-h-40 overflow-y-auto">
                    {players.map((p) => (
                      <button
                        key={p.id}
                        className="w-full text-left px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-700"
                        onClick={() => { setNewInjury({ ...newInjury, player_id: p.id }); setPlayerSearch(p.full_name ?? `#${p.id}`); setPlayers([]); }}
                      >
                        {p.full_name ?? `#${p.id}`} {p.team_abbreviation ? `(${p.team_abbreviation})` : ""}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {/* Status + type row */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-400 block mb-1">伤病状态</label>
                  <select
                    value={newInjury.injury_status}
                    onChange={(e) => setNewInjury({ ...newInjury, injury_status: e.target.value })}
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200"
                  >
                    {["Out", "Doubtful", "Questionable", "Probable", "Available"].map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">伤病类型</label>
                  <input
                    type="text"
                    value={newInjury.injury_type}
                    onChange={(e) => setNewInjury({ ...newInjury, injury_type: e.target.value })}
                    placeholder="如: 左脚踝扭伤"
                    className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200"
                  />
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">补充描述</label>
                <input
                  type="text"
                  value={newInjury.description}
                  onChange={(e) => setNewInjury({ ...newInjury, description: e.target.value })}
                  placeholder="如: 预计缺席2-3周"
                  className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-200"
                />
              </div>
              <button
                className="bg-purple-600 hover:bg-purple-700 text-white text-sm px-4 py-1.5 rounded transition-colors disabled:opacity-50"
                onClick={handleCreateInjury}
                disabled={creating || newInjury.player_id === 0}
              >
                {creating ? "创建中..." : "确认新增"}
              </button>
            </div>
          </div>
        )}

        {injuriesLoading ? (
          <div className="text-center py-4">
            <div className="inline-block w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : sortedInjuries.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-3xl mb-2">✅</div>
            <p className="text-gray-400">所有伤病已审核</p>
            <p className="text-xs text-gray-600 mt-1">系统自动拉取的伤病建议稿会显示在这里</p>
          </div>
        ) : (
          <div className="space-y-2">
            {/* Pending count banner */}
            <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg text-sm text-amber-400 mb-3">
              ⚠️ 有 {injuries.length} 条伤病待审核。请逐条确认、修改或删除，或使用"一键确认"。
            </div>

            {sortedInjuries.map((injury) => (
              <div
                key={injury.id}
                className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800 transition-colors"
              >
                {/* Player info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span
                      className="text-gray-200 font-medium cursor-pointer hover:text-blue-400"
                      onClick={() => navigate(`/players/${injury.player_id}`)}
                    >
                      {injury.player_name ?? `Player #${injury.player_id}`}
                    </span>
                    {injury.team_abbreviation && (
                      <span className="text-xs text-gray-500">{injury.team_abbreviation}</span>
                    )}
                    {/* Confirmation status badge */}
                    <span className={`text-xs px-1.5 py-0.5 rounded ${confirmationBadge(injury.confirmation ?? "pending")}`}>
                      {confirmationLabel(injury.confirmation ?? "pending")}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 mt-0.5">
                    {editingId === injury.id ? (
                      <select
                        value={editStatus}
                        onChange={(e) => setEditStatus(e.target.value)}
                        className="bg-gray-700 border border-gray-600 rounded px-2 py-0.5 text-xs text-gray-200"
                      >
                        <option value="">选择状态</option>
                        {["Out", "Doubtful", "Questionable", "Probable", "Available"].map((s) => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    ) : (
                      <span
                        className={`text-xs px-1.5 py-0.5 rounded ${
                          injury.injury_status === "Out"
                            ? "bg-red-500/20 text-red-400"
                            : injury.injury_status === "Doubtful"
                            ? "bg-amber-500/20 text-amber-400"
                            : "bg-blue-500/20 text-blue-400"
                        }`}
                      >
                        {injury.injury_status}
                      </span>
                    )}
                    {injury.injury_type && (
                      <span className="text-xs text-gray-500">{injury.injury_type}</span>
                    )}
                    {injury.description && (
                      <span className="text-xs text-gray-600 truncate max-w-xs">
                        — {injury.description}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-600 mt-0.5">
                    来源: {injury.source === "nba_api" ? "NBA API" : "手动录入"}
                    {" · "}
                    {new Date(injury.reported_date).toLocaleDateString("zh-CN")}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1.5">
                  {editingId === injury.id ? (
                    <>
                      <button
                        className="text-xs bg-emerald-600 hover:bg-emerald-700 text-white px-2 py-1 rounded"
                        onClick={() => handleEdit(injury.id)}
                      >
                        保存
                      </button>
                      <button
                        className="text-xs bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 rounded"
                        onClick={() => setEditingId(null)}
                      >
                        取消
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        className="text-xs bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-400 px-2 py-1 rounded transition-colors"
                        onClick={() => handleInjuryAction(injury, "confirm")}
                      >
                        确认
                      </button>
                      <button
                        className="text-xs bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 px-2 py-1 rounded transition-colors"
                        onClick={() => { setEditingId(injury.id); setEditStatus(injury.injury_status); }}
                      >
                        修改
                      </button>
                      <button
                        className="text-xs bg-red-600/20 hover:bg-red-600/40 text-red-400 px-2 py-1 rounded transition-colors"
                        onClick={() => handleInjuryAction(injury, "delete")}
                      >
                        删除
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
