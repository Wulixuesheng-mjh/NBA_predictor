/** ECharts gauge chart for prediction accuracy display. */

import ReactECharts from "echarts-for-react";

interface Props {
  winAccuracy: number;
  spreadWithin4: number; // fraction
  totalWithin8: number; // fraction
}

export default function AccuracyGauge({ winAccuracy, spreadWithin4, totalWithin8 }: Props) {
  const option = {
    series: [
      {
        type: "gauge",
        startAngle: 210,
        endAngle: -30,
        center: ["50%", "55%"],
        radius: "80%",
        min: 0,
        max: 1,
        axisLine: {
          show: true,
          lineStyle: {
            width: 16,
            color: [
              [0.5, "#ef4444"],
              [0.6, "#f59e0b"],
              [0.75, "#3b82f6"],
              [1, "#10b981"],
            ],
          },
        },
        axisTick: { show: false },
        splitLine: { show: false },
        axisLabel: {
          show: true,
          formatter: (v: number) => `${(v * 100).toFixed(0)}%`,
          color: "#9ca3af",
          fontSize: 10,
        },
        pointer: {
          length: "60%",
          width: 6,
          itemStyle: { color: "#e5e7eb" },
        },
        detail: {
          valueAnimation: true,
          formatter: (v: number) => `${(v * 100).toFixed(1)}%`,
          color: "#e5e7eb",
          fontSize: 20,
          fontWeight: "bold",
          offsetCenter: [0, "70%"],
        },
        data: [{ value: winAccuracy, name: "胜率" }],
      },
    ],
  };

  return (
    <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
      <h4 className="text-sm font-medium text-gray-300 mb-1">预测准确率</h4>
      <ReactECharts option={option} style={{ height: 220 }} />
      <div className="flex justify-around text-center mt-1">
        <div>
          <div className="text-xs text-gray-500">让分 ±4</div>
          <div className="text-sm font-mono text-gray-300">
            {(spreadWithin4 * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="text-xs text-gray-500">总分 ±8</div>
          <div className="text-sm font-mono text-gray-300">
            {(totalWithin8 * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
