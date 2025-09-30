import React from "react";
import { Bar } from "@ant-design/plots";
import type { SentenceResult } from "../types";

interface Props {
  data: SentenceResult[];
}

const ChartTab: React.FC<Props> = ({ data }) => {
  const statsData = data.flatMap((d) => d.role_answers);
  const chartData = statsData.reduce<Record<string, number>>((acc, cur) => {
    const key = `${cur.event_type} - ${cur.role}`;
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const plotData = Object.entries(chartData).map(([key, count]) => ({
    role: key,
    count,
  }));

  const config = {
    data: plotData,
    xField: "count",
    yField: "role",
    seriesField: "role",
    legend: false,
  };

  return <Bar {...config} />;
};

export default ChartTab;
