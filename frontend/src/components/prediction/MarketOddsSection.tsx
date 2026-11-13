/** 体彩胜负赔率 vs 模型概率 — market implied probability comparison. */

import type { MarketOdds } from "../../types";

interface Props {
  market: MarketOdds;
  homeName: string;
  awayName: string;
}

function Row({
  label,
  odds,
  implied,
  model,
  edge,
  highlight,
}: {
  label: string;
  odds: number;
  implied: number;
  model: number;
  edge: number;
  highlight: boolean;
}) {
  return (
    <div
      className={`grid grid-cols-5 items-center gap-2 px-3 py-2 rounded-lg ${
        highlight ? "bg-emerald-500/10 border border-emerald-500/30" : "bg-gray-900/40"
      }`}
    >
      <span className="text-sm text-gray-300">{label}</span>
      <span className="text-sm font-mono text-gray-200 text-right">{odds.toFixed(2)}</span>
      <span className="text-sm font-mono text-gray-400 text-right">{(implied * 100).toFixed(1)}%</span>
      <span className="text-sm font-mono text-blue-300 text-right">{(model * 100).toFixed(1)}%</span>
      <span
        className={`text-sm font-mono text-right ${
          edge >= 0.03 ? "text-emerald-400" : edge <= -0.03 ? "text-red-400" : "text-gray-500"
        }`}
      >
        {edge >= 0 ? "+" : ""}
        {(edge * 100).toFixed(1)}pp
      </span>
    </div>
  );
}

export default function MarketOddsSection({ market, homeName, awayName }: Props) {
  const valueText =
    market.value_side === "none"
      ? "模型与市场分歧很小，无明显价值"
      : market.value_side === "home"
      ? `模型认为主队（${homeName}）被低估`
      : `模型认为客队（${awayName}）被低估`;

  // 抽水 = 1/主胜 + 1/客胜 - 1
  const margin = (market.overround - 1) * 100;

  return (
    <div className="bg-gray-800/60 rounded-xl border border-amber-500/30 overflow-hidden">
      <div className="px-5 py-3 flex items-center justify-between border-b border-gray-700/50">
        <div>
          <h3 className="text-sm font-semibold text-gray-200">胜负赔率 · 市场对比</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            录入赔率 主 {market.home_odds.toFixed(2)} / 客 {market.away_odds.toFixed(2)} · 抽水约{" "}
            {margin.toFixed(1)}%
          </p>
        </div>
        <span className="text-xs bg-amber-600/20 text-amber-400 px-2 py-0.5 rounded">
          对比模型 {market.model_source === "mlp" ? "MLP" : "XGBoost"}
        </span>
      </div>

      <div className="p-5">
        <div className="grid grid-cols-5 gap-2 px-3 mb-1 text-xs text-gray-500">
          <span>方向</span>
          <span className="text-right">体彩赔率</span>
          <span className="text-right">市场隐含概率</span>
          <span className="text-right">模型概率</span>
          <span className="text-right">模型优势</span>
        </div>
        <Row
          label={`主胜 · ${homeName}`}
          odds={market.home_odds}
          implied={market.implied_home}
          model={market.model_home}
          edge={market.edge_home}
          highlight={market.value_side === "home"}
        />
        <div className="mt-1">
          <Row
            label={`客胜 · ${awayName}`}
            odds={market.away_odds}
            implied={market.implied_away}
            model={1 - market.model_home}
            edge={market.edge_away}
            highlight={market.value_side === "away"}
          />
        </div>

        <div
          className={`mt-4 p-3 rounded-lg text-sm ${
            market.value_side === "none"
              ? "bg-gray-900/50 text-gray-400"
              : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-300"
          }`}
        >
          {market.value_side === "none" ? "⚖️ " : "💡 "}
          {valueText}
          {market.value_side !== "none" && (
            <span className="text-emerald-400/80">
              （高出市场 {(market.value_edge * 100).toFixed(1)} 个百分点）
            </span>
          )}
        </div>

        <p className="mt-3 text-xs text-gray-600">
          市场隐含概率已剔除抽水（归一化到 100%）；「模型优势」为正表示模型比市场更看好该方向。模型仍受数据质量影响，仅供参考。
        </p>
      </div>
    </div>
  );
}
