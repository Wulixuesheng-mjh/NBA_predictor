/** Collapsible SHAP explanation section — shows why the model made this prediction. */

import { useState } from "react";
import type { ExplanationOutput } from "../../types";

interface Props {
  explanation: ExplanationOutput;
}

export default function ExplanationSection({ explanation }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (!explanation.top_features || explanation.top_features.length === 0) {
    return null;
  }

  // Take top 3 features for summary display
  const top3 = explanation.top_features.slice(0, 3);

  return (
    <div className="bg-gray-800/60 rounded-xl border border-gray-700/50 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-3 flex items-center justify-between hover:bg-gray-800/40 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-300">为什么这个预测？</span>
          <span className="text-xs text-gray-500">XGBoost SHAP 解释</span>
        </div>
        <span className={`text-gray-500 transition-transform ${expanded ? "rotate-180" : ""}`}>
          ▼
        </span>
      </button>

      {/* Collapsed: quick summary of top 3 */}
      {!expanded && (
        <div className="px-5 pb-3 space-y-2">
          {top3.map((f) => (
            <div key={f.feature} className="flex items-center gap-2 text-sm">
              <span className={f.direction === "favors_home" ? "text-emerald-400" : "text-red-400"}>
                {f.direction === "favors_home" ? "↑" : "↓"}
              </span>
              <span className="text-gray-300 flex-1">{f.display_name}</span>
              <span className={`font-mono text-xs font-semibold ${
                f.contribution === "positive" ? "text-emerald-400" : "text-red-400"
              }`}>
                {f.contribution === "positive" ? "+" : ""}{f.shap_value.toFixed(3)} log-odds
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Expanded: full detail with contribution bars */}
      {expanded && (
        <div className="px-5 pb-4 space-y-2">
          {explanation.top_features.map((f) => {
            const barWidth = Math.min(Math.abs(f.shap_value) * 100 * 3, 100); // scale for visibility
            return (
              <div key={f.feature} className="space-y-1">
                <div className="flex items-center gap-2 text-sm">
                  <span className={f.direction === "favors_home" ? "text-emerald-400" : "text-red-400"}>
                    {f.direction === "favors_home" ? "↑" : "↓"}
                  </span>
                  <span className="text-gray-300 flex-1">{f.display_name}</span>
                  <span className={`font-mono text-xs ${
                    f.contribution === "positive" ? "text-emerald-400" : "text-red-400"
                  }`}>
                    {f.contribution === "positive" ? "+" : ""}{f.shap_value.toFixed(3)} log-odds
                  </span>
                </div>
                <div className="h-1.5 bg-gray-900 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      f.direction === "favors_home" ? "bg-emerald-500" : "bg-red-500"
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
          <p className="text-xs text-gray-600 mt-3">
            基于 {explanation.explained_model} 模型，SHAP 值为胜负模型 log-odds 空间的边际贡献。
            正值 = 增加主胜概率，负值 = 降低主胜概率。
          </p>
        </div>
      )}
    </div>
  );
}
