import React from "react";
import { Heatmap } from "@ant-design/plots";
import type { SentenceResult } from "../../types";

interface Props {
  data: SentenceResult[];
}

const EventRoleHeatmap: React.FC<Props> = ({ data }) => {
  const statsData = data.flatMap((d) => d.role_answers);
  const countMap: Record<string, number> = {};
  statsData.forEach((item) => {
    if (!item.answer.includes("none")) {
      const key = `${item.event_type}-${item.role}`;
      countMap[key] = (countMap[key] || 0) + 1;
    }
  });

  const chartData = Object.entries(countMap).map(([key, count]) => {
    const [event_type, role] = key.split("-");
    return { event_type, role, value: count };
  });

  const config = {
    data: chartData,
    xField: "event_type",
    yField: "role",
    colorField: "value",
    tooltip: { fields: ["event_type", "role", "value"] },
    height: 400,
  };

  return (
    <div>
      <h3>Tương quan Event ↔ Role</h3>
      <Heatmap {...config} />
    </div>
  );
};

export default EventRoleHeatmap;
