/** ECharts line chart — sliding-window accuracy trends. */

import ReactECharts from "echarts-for-react";
import type { AccuracyTrendResponse } from "../../types";

interface Props {
  data: AccuracyTrendResponse;
}

export default function AccuracyTrendChart({ data }: Props) {
  if (!data.trends || data.trends.length === 0) {
    return (
      <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
        <h4 className="text-sm font-medium text-gray-300 mb-2">准确率趋势</h4>
        <p className="text-gray-500 text-center py-8 text-sm">
          暂无趋势数据。预测评估后自动生成。
        </p>
      </div>
    );
  }

  const dates = data.trends.map((t) => t.date);
  const winAcc = data.trends.map((t) =>
    t.win_accuracy != null ? +(t.win_accuracy * 100).toFixed(1) : null,
  );
  const spreadMae = data.trends.map((t) => t.spread_mae);
  const totalMae = data.trends.map((t) => t.total_mae);
  const samples = data.trends.map((t) => t.sample_count);

  const option = {
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(20, 20, 30, 0.95)",
      borderColor: "#374151",
      textStyle: { color: "#e5e7eb", fontSize: 12 },
      formatter: (params: any[]) => {
        if (!params || params.length === 0) return "";
        const date = params[0].axisValue;
        let html = `<div style="font-weight:600;margin-bottom:4px">${date}</div>`;
        for (const p of params) {
          if (p.value == null) continue;
          const val =
            p.seriesName === "胜负准确率"
              ? `${p.value}%`
              : p.value.toFixed(1);
          html += `<div style="display:flex;align-items:center;gap:6px">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color}"></span>
            <span>${p.seriesName}: <b>${val}</b></span>
          </div>`;
        }
        // Sample count
        const idx = params[0]?.dataIndex;
        if (idx != null && samples[idx] > 0) {
          html += `<div style="color:#6b7280;font-size:10px;margin-top:2px">${samples[idx]} 场预测</div>`;
        }
        return html;
      },
    },
    legend: {
      data: ["胜负准确率", "让分MAE", "总分MAE"],
      bottom: 0,
      textStyle: { color: "#9ca3af", fontSize: 11 },
      itemGap: 16,
    },
    grid: {
      left: "3%",
      right: "4%",
      top: "8%",
      bottom: "12%",
      containLabel: true,
    },
    xAxis: {
      type: "category",
      data: dates,
      axisLine: { lineStyle: { color: "#374151" } },
      axisTick: { show: false },
      axisLabel: {
        color: "#6b7280",
        fontSize: 10,
        formatter: (v: string) => v.slice(5), // MM-DD
        interval: Math.max(0, Math.floor(dates.length / 8) - 1),
      },
    },
    yAxis: [
      {
        type: "value",
        name: "胜负准确率 (%)",
        min: 0,
        max: 100,
        axisLabel: { color: "#6b7280", fontSize: 10, formatter: "{value}%" },
        splitLine: { lineStyle: { color: "#1f2937" } },
        nameTextStyle: { color: "#6b7280", fontSize: 10 },
      },
      {
        type: "value",
        name: "MAE (分)",
        axisLabel: { color: "#6b7280", fontSize: 10 },
        splitLine: { show: false },
        nameTextStyle: { color: "#6b7280", fontSize: 10 },
      },
    ],
    series: [
      {
        name: "胜负准确率",
        type: "line",
        data: winAcc,
        yAxisIndex: 0,
        smooth: true,
        symbol: "none",
        lineStyle: { color: "#3b82f6", width: 2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(59, 130, 246, 0.3)" },
              { offset: 1, color: "rgba(59, 130, 246, 0.02)" },
            ],
          },
        },
        connectNulls: false,
      },
      {
        name: "让分MAE",
        type: "line",
        data: spreadMae,
        yAxisIndex: 1,
        smooth: true,
        symbol: "none",
        lineStyle: { color: "#f59e0b", width: 1.5, type: "dashed" },
        connectNulls: false,
      },
      {
        name: "总分MAE",
        type: "line",
        data: totalMae,
        yAxisIndex: 1,
        smooth: true,
        symbol: "none",
        lineStyle: { color: "#ef4444", width: 1.5, type: "dashed" },
        connectNulls: false,
      },
    ],
  };

  // Summary stats for the cards below the chart
  const { overall } = data;
  const winPct = (overall.win_accuracy * 100).toFixed(1);
  const spreadText =
    overall.spread_mae != null ? overall.spread_mae.toFixed(1) : "—";
  const totalText =
    overall.total_mae != null ? overall.total_mae.toFixed(1) : "—";

  return (
    <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-300">
          准确率趋势（{data.trends[0]?.window_days ?? 7}天滑动窗口）
        </h4>
        <span className="text-xs text-gray-500">
          共 {overall.total_evaluated} 场已评估
        </span>
      </div>
      <ReactECharts option={option} style={{ height: 300 }} />
      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-3 mt-3">
        {[
          ["胜负准确率", winPct + "%", "text-blue-400"],
          ["让分MAE", spreadText, "text-amber-400"],
          ["总分MAE", totalText, "text-red-400"],
        ].map(([label, value, color]) => (
          <div
            key={label as string}
            className="bg-gray-900/50 rounded-lg p-2 text-center"
          >
            <div className="text-xs text-gray-500">{label}</div>
            <div className={`text-lg font-bold font-mono ${color}`}>
              {value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
