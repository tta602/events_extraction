import React from "react";
import { Bar } from "@ant-design/plots";
import type { SentenceResult } from "../../types";

interface Props {
  data: SentenceResult[];
}

const RoleBarChart: React.FC<Props> = ({ data }) => {
  const statsData = data.flatMap((d) => d.role_answers);
  const plotData = statsData.map((item) => ({
    event_type: item.event_type,
    role: item.role,
  }));

  const countMap: Record<string, number> = {};
  plotData.forEach((item) => {
    const key = `${item.event_type}-${item.role}`;
    countMap[key] = (countMap[key] || 0) + 1;
  });

  const chartData = Object.entries(countMap).map(([key, count]) => {
    const [event_type, role] = key.split("-");
    return { event_type, role, count };
  });

  const config = {
    data: chartData,
    xField: "count",
    yField: "role",
    seriesField: "event_type",
    isGroup: true,
    legend: { position: "top-left" },
    height: 400,
  };

  return (
    <div>
      <h3>Vai trò trong từng Event</h3>
      <Bar {...config} />
    </div>
  );
};

export default RoleBarChart;
