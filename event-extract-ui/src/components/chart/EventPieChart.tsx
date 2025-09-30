import React from "react";
import { Pie } from "@ant-design/plots";
import type { SentenceResult } from "../../types";

interface Props {
  data: SentenceResult[];
}

const EventPieChart: React.FC<Props> = ({ data }) => {
  const events = data.flatMap((d) => d.role_answers.map((r) => r.event_type));
  const countMap: Record<string, number> = {};
  events.forEach((e) => {
    if (!e.includes("none")) {
      countMap[e] = (countMap[e] || 0) + 1;
    }
  });

  const chartData = Object.entries(countMap).map(([type, count]) => ({
    type,
    value: count,
  }));

  const config = {
    data: chartData,
    angleField: "value",
    colorField: "type",
    radius: 0.9,
    label: { type: "inner", offset: "-30%", content: "{value}", style: { fontSize: 14 } },
    interactions: [{ type: "element-active" }],
  };

  return (
    <div>
      <h3>Phân bố Event Types</h3>
      <Pie {...config} />
    </div>
  );
};

export default EventPieChart;
