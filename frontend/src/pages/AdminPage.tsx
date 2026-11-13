/** AdminPage — system stats, API key, collection & training triggers.

 * The injury review workbench was removed: injuries are now entered manually
 * on the HomePage (see InjuryEntry), so there is no auto-pull pending queue.
 */

import { useState } from "react";
import { useAdminStats } from "../hooks/useAdminStats";
import { triggerCollection, triggerModelTraining, syncRosters } from "../api/admin";

export default function AdminPage() {
  const { stats, loading: statsLoading, refresh: refreshStats } = useAdminStats();

  const [apiKey, setApiKey] = useState(() => localStorage.getItem("admin_api_key") || "");
  const [showKeyInput, setShowKeyInput] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [syncingRosters, setSyncingRosters] = useState(false);
  const [rosterMsg, setRosterMsg] = useState("");
  const [training, setTraining] = useState(false);
  const [trainingMsg, setTrainingMsg] = useState("");

  const handleSaveApiKey = () => {
    localStorage.setItem("admin_api_key", apiKey);
    setShowKeyInput(false);
    refreshStats(); // re-fetch with new key
  };

  const handleClearApiKey = () => {
    localStorage.removeItem("admin_api_key");
    setApiKey("");
    setShowKeyInput(false);
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

  const handleSyncRosters = async () => {
    setSyncingRosters(true);
    setRosterMsg("正在从 nba_api 拉取最新阵容...");
    try {
      const result = await syncRosters();
      setRosterMsg(`阵容已更新：写入 ${result.players_synced} 名球员`);
      await refreshStats();
    } catch (e: any) {
      setRosterMsg(`阵容更新失败: ${e.message}`);
    } finally {
      setSyncingRosters(false);
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
          <button className="btn-secondary" onClick={handleSyncRosters} disabled={syncingRosters}>
            {syncingRosters ? "更新阵容中..." : "更新本赛季阵容"}
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
        {rosterMsg && (
          <p className="mt-2 text-sm text-gray-400">{rosterMsg}</p>
        )}
      </div>
    </div>
  );
}
