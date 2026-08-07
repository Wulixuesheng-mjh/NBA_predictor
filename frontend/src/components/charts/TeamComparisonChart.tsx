/** ECharts radar/spider chart comparing two teams across key metrics. */

import ReactECharts from "echarts-for-react";
import type { TeamDetail } from "../../types";

interface Props {
  homeTeam: TeamDetail | null;
  awayTeam: TeamDetail | null;
}

export default function TeamComparisonChart({ homeTeam, awayTeam }: Props) {
  if (!homeTeam || !awayTeam) return null;

  const indicators = [
    { name: "胜率", max: 1 },
    { name: "场均得分", max: 130 },
    { name: "场均失分", max: 130 },
    { name: "净胜分", max: 15 },
  ];

  const homeName = homeTeam.abbreviation;
  const awayName = awayTeam.abbreviation;

  // Parse recent record for win rate
  const parseWinRate = (record: string | null) => {
    if (!record) return 0;
    const parts = record.split("-");
    if (parts.length === 2) {
      const wins = parseInt(parts[0]);
      const total = wins + parseInt(parts[1]);
      return total > 0 ? wins / total : 0;
    }
    return 0;
  };

  const option = {
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(20, 20, 30, 0.9)",
      borderColor: "#333",
      textStyle: { color: "#e5e7eb" },
    },
    legend: {
      data: [homeName, awayName],
      bottom: 0,
      textStyle: { color: "#9ca3af" },
    },
    radar: {
      indicator: indicators,
      center: ["50%", "50%"],
      radius: "65%",
      axisName: { color: "#9ca3af", fontSize: 11 },
      splitArea: {
        areaStyle: { color: ["#111827", "#1f2937"] },
      },
      splitLine: { lineStyle: { color: "#374151" } },
      axisLine: { lineStyle: { color: "#374151" } },
    },
    series: [
      {
        type: "radar",
        data: [
          {
            name: homeName,
            value: [
              parseWinRate(homeTeam.recent_record),
              homeTeam.avg_pts ?? 0,
              homeTeam.avg_pts_allowed ?? 0,
              homeTeam.net_rating ?? 0,
            ],
            itemStyle: { color: "#3b82f6" },
            areaStyle: { color: "rgba(59, 130, 246, 0.2)" },
            lineStyle: { color: "#3b82f6" },
          },
          {
            name: awayName,
            value: [
              parseWinRate(awayTeam.recent_record),
              awayTeam.avg_pts ?? 0,
              awayTeam.avg_pts_allowed ?? 0,
              awayTeam.net_rating ?? 0,
            ],
            itemStyle: { color: "#ef4444" },
            areaStyle: { color: "rgba(239, 68, 68, 0.2)" },
            lineStyle: { color: "#ef4444" },
          },
        ],
      },
    ],
  };

  return (
    <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
      <h4 className="text-sm font-medium text-gray-300 mb-2">球队对比</h4>
      <ReactECharts option={option} style={{ height: 280 }} />
    </div>
  );
}
