/** Consensus indicator between XGBoost and MLP predictions. */

import type { ConsensusOutput } from "../../types";

interface Props {
  consensus: ConsensusOutput;
}

function AgreementIcon({ agree, label }: { agree: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={agree ? "text-emerald-400" : "text-red-400"}>
        {agree ? "✓" : "✗"}
      </span>
      <span className="text-xs text-gray-400">{label}</span>
    </div>
  );
}

export default function ConsensusBadge({ consensus }: Props) {
  const scorePct = (consensus.agreement_score * 100).toFixed(0);
  const scoreColor =
    consensus.agreement_score >= 0.67
      ? "text-emerald-400"
      : consensus.agreement_score >= 0.34
      ? "text-amber-400"
      : "text-red-400";

  return (
    <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-gray-300">模型一致性</h4>
        <span className={`text-lg font-bold ${scoreColor}`}>{scorePct}%</span>
      </div>
      <div className="flex gap-4">
        <AgreementIcon agree={consensus.win_agree} label="胜负方向" />
        <AgreementIcon agree={consensus.spread_direction_agree} label="让分方向" />
        <AgreementIcon agree={consensus.total_agree} label="大小分" />
      </div>
      {consensus.agreement_score < 0.5 && (
        <p className="text-xs text-amber-400 mt-2">
          ⚠️ 两模型分歧较大，建议谨慎参考
        </p>
      )}
    </div>
  );
}
