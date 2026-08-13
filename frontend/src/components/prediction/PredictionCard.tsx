/** Dual-model prediction result card. */

import type { ModelPredictionOutput, ConfidenceOutput } from "../../types";

interface Props {
  modelName: string;
  modelLabel: string;
  prediction: ModelPredictionOutput;
  recommended: boolean;
  confidence?: ConfidenceOutput | null;
}

const CONFIDENCE_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  high: { bg: "bg-emerald-500/10 border-emerald-500/30", text: "text-emerald-400", dot: "bg-emerald-400" },
  medium: { bg: "bg-amber-500/10 border-amber-500/30", text: "text-amber-400", dot: "bg-amber-400" },
  low: { bg: "bg-orange-500/10 border-orange-500/30", text: "text-orange-400", dot: "bg-orange-400" },
  not_recommended: { bg: "bg-red-500/10 border-red-500/30", text: "text-red-400", dot: "bg-red-400" },
};

function WinBar({ prob, isHome }: { prob: number; isHome: boolean }) {
  const pct = (prob * 100).toFixed(1);
  const barWidth = Math.max(prob * 100, 8); // minimum 8% for label visibility
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-8">{isHome ? "主胜" : "客胜"}</span>
      <div className="flex-1 h-5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-600 rounded-full flex items-center justify-end pr-2 transition-all duration-500"
          style={{ width: `${barWidth}%` }}
        >
          <span className="text-xs font-mono text-white">{pct}%</span>
        </div>
      </div>
    </div>
  );
}

export default function PredictionCard({ modelName: _modelName, modelLabel, prediction, recommended, confidence }: Props) {
  const winHome = prediction.win_prob_home > 0.5;
  const confStyle = confidence ? CONFIDENCE_STYLES[confidence.level] ?? CONFIDENCE_STYLES.not_recommended : null;

  return (
    <div
      className={`bg-gray-800/60 rounded-xl p-5 border-2 transition-colors ${
        recommended ? "border-blue-500/50" : "border-gray-700/50"
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
          {modelLabel}
        </h3>
        <div className="flex items-center gap-2">
          {recommended && (
            <span className="text-xs bg-blue-600/20 text-blue-400 px-2 py-0.5 rounded">
              主力模型
            </span>
          )}
          {confStyle && confidence && (
            <span className={`text-xs px-2 py-0.5 rounded border ${confStyle.bg} ${confStyle.text}`}
                  title={`置信度评分: ${confidence.score.toFixed(0)}/100`}>
              <span className={`inline-block w-1.5 h-1.5 rounded-full ${confStyle.dot} mr-1`} />
              {confidence.label_cn}
            </span>
          )}
          <span className="text-xs text-gray-500 font-mono">{prediction.model_version}</span>
        </div>
      </div>

      {/* Win/Loss */}
      <div className="space-y-2 mb-4">
        <h4 className="text-xs text-gray-500 uppercase">胜负预测</h4>
        <WinBar prob={prediction.win_prob_home} isHome />
        <div className="text-center text-sm mt-1">
          <span className={winHome ? "text-emerald-400 font-semibold" : "text-red-400 font-semibold"}>
            推荐: {winHome ? "主胜" : "客胜"}
          </span>
          <span className="text-gray-500 text-xs ml-1">
            (信心: {((prediction.spread_confidence ?? 0) * 100).toFixed(0)}%)
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Spread */}
        <div className="bg-gray-900/50 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500 mb-1">让分预测</div>
          <div className={`text-xl font-bold ${prediction.predicted_spread !== null && prediction.predicted_spread > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {prediction.predicted_spread !== null
              ? `${prediction.predicted_spread > 0 ? '+' : ''}${prediction.predicted_spread.toFixed(1)}`
              : '—'}
          </div>
          <div className="text-xs text-gray-600 mt-1">
            {prediction.predicted_spread !== null && prediction.predicted_spread > 0
              ? '主队赢分'
              : prediction.predicted_spread !== null
              ? '客队赢分'
              : ''}
          </div>
        </div>

        {/* Total */}
        <div className="bg-gray-900/50 rounded-lg p-3 text-center">
          <div className="text-xs text-gray-500 mb-1">总分预测</div>
          <div className="text-xl font-bold text-amber-400">
            {prediction.predicted_total !== null
              ? prediction.predicted_total.toFixed(1)
              : '—'}
          </div>
          <div className="text-xs text-gray-600 mt-1">
            {prediction.predicted_total !== null
              ? prediction.predicted_total > 225
                ? '推荐大分 ↑'
                : '推荐小分 ↓'
              : ''}
          </div>
        </div>
      </div>
    </div>
  );
}
