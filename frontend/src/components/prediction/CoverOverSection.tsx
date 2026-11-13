/** 盘口 (让分胜负 / 大小分) binary prediction section. */

import { useState } from "react";
import type { CoverOverOutput, CoverOverExplanationFeature } from "../../types";

interface Props {
  coverOver: CoverOverOutput;
}

function DirectionBar({ prob, label, active }: { prob: number; label: string; active: boolean }) {
  const pct = (prob * 100).toFixed(1);
  const width = Math.max(prob * 100, 8);
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-20 shrink-0">{label}</span>
      <div className="flex-1 h-5 bg-gray-900 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full flex items-center justify-end pr-2 transition-all duration-500 ${
            active ? "bg-emerald-500" : "bg-gray-600"
          }`}
          style={{ width: `${width}%` }}
        >
          <span className="text-xs font-mono text-white">{pct}%</span>
        </div>
      </div>
    </div>
  );
}

function FeatureList({ features }: { features: CoverOverExplanationFeature[] | null }) {
  if (!features || features.length === 0) return <p className="text-xs text-gray-600">无可解释特征</p>;
  return (
    <div className="space-y-1">
      {features.map((f) => (
        <div key={f.feature} className="flex items-center gap-2 text-xs">
          <span className={f.contribution === "positive" ? "text-emerald-400" : "text-red-400"}>
            {f.contribution === "positive" ? "↑" : "↓"}
          </span>
          <span className="text-gray-300 flex-1">{f.display_name}</span>
          <span className="font-mono text-gray-500">
            {f.contribution === "positive" ? "+" : ""}{f.shap_value.toFixed(3)} log-odds
          </span>
        </div>
      ))}
    </div>
  );
}

export default function CoverOverSection({ coverOver }: Props) {
  const [expanded, setExpanded] = useState(false);
  const coverHome = coverOver.cover_prob_home >= 0.5;
  const over = coverOver.over_prob >= 0.5;
  const hasExplanation =
    (coverOver.explanation?.cover?.length ?? 0) > 0 || (coverOver.explanation?.over?.length ?? 0) > 0;

  return (
    <div className="bg-gray-800/60 rounded-xl border border-purple-500/30 overflow-hidden">
      <div className="px-5 py-3 flex items-center justify-between border-b border-gray-700/50">
        <div>
          <h3 className="text-sm font-semibold text-gray-200">盘口预测 · 让分胜负 / 大小分</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            录入盘口：让分{" "}
            {coverOver.spread_line != null
              ? `${coverOver.spread_line > 0 ? "+" : ""}${coverOver.spread_line}`
              : "—"}{" "}
            · 大小分 {coverOver.total_line ?? "—"}
          </p>
        </div>
        <span className="text-xs bg-purple-600/20 text-purple-400 px-2 py-0.5 rounded">
          {coverOver.calibrated ? `样本外残差 · n=${coverOver.residual_samples}` : "未校验分布"}
        </span>
      </div>

      <div className="p-5 grid grid-cols-2 gap-6">
        {/* 让分胜负 */}
        <div>
          <div className="text-xs text-gray-500 mb-2">让分胜负（主队赢盘 vs 客队赢盘）</div>
          <div className="mb-2">
            <span className={`text-lg font-bold ${coverHome ? "text-emerald-400" : "text-red-400"}`}>
              推荐: {coverOver.cover_direction_cn}
            </span>
          </div>
          <DirectionBar prob={coverOver.cover_prob_home} label="主队赢盘" active={coverHome} />
          <div className="mt-1">
            <DirectionBar prob={coverOver.cover_prob_away} label="客队赢盘" active={!coverHome} />
          </div>
          {coverOver.cover_push_prob > 0 && <p className="text-xs text-gray-500 mt-1">走盘 {(coverOver.cover_push_prob * 100).toFixed(1)}%</p>}
        </div>

        {/* 大小分 */}
        <div>
          <div className="text-xs text-gray-500 mb-2">大小分（总分 vs 盘口）</div>
          <div className="mb-2">
            <span className={`text-lg font-bold ${over ? "text-amber-400" : "text-sky-400"}`}>
              推荐: {coverOver.over_direction_cn}
            </span>
          </div>
          <DirectionBar prob={coverOver.over_prob} label="大分" active={over} />
          <div className="mt-1">
            <DirectionBar prob={coverOver.under_prob} label="小分" active={!over} />
          </div>
          {coverOver.total_push_prob > 0 && <p className="text-xs text-gray-500 mt-1">走盘 {(coverOver.total_push_prob * 100).toFixed(1)}%</p>}
        </div>
      </div>

      {hasExplanation && (
        <div className="border-t border-gray-700/50">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-5 py-2.5 flex items-center justify-between hover:bg-gray-800/40 transition-colors"
          >
            <span className="text-xs text-gray-400">盘口 SHAP 解释</span>
            <span className={`text-gray-500 text-xs transition-transform ${expanded ? "rotate-180" : ""}`}>▼</span>
          </button>
          {expanded && (
            <div className="px-5 pb-4 grid grid-cols-2 gap-6">
              <div>
                <div className="text-xs text-gray-500 mb-2">让分胜负关键因素</div>
                <FeatureList features={coverOver.explanation?.cover ?? null} />
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-2">大小分关键因素</div>
                <FeatureList features={coverOver.explanation?.over ?? null} />
              </div>
            </div>
          )}
        </div>
      )}

      <div className="px-5 py-2 bg-gray-900/40 border-t border-gray-700/50">
        <p className="text-xs text-gray-600">
          {coverOver.calibrated
            ? "概率来自最近样本外残差分布，会随输入盘口单调变化。"
            : "当前样本外残差不足 200 条，概率使用未校验的正态近似；达到门槛后自动切换。"}
        </p>
      </div>
    </div>
  );
}
