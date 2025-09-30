import React from "react";
import type { SentenceResult } from "../types";
import EventPieChart from "./chart/EventPieChart";
import RoleBarChart from "./chart/RoleBarChart";
import EventRoleHeatmap from "./chart/EventRoleHeatmap";
import EventRoleGraph from "./chart/EventRoleGraph";

interface Props {
  data: SentenceResult[];
}

const ChartTab: React.FC<Props> = ({ data }) => {
  return (
    <div style={{ display: "grid", gap: 32 }}>
      <EventPieChart data={data} />
      {/* <RoleBarChart data={data} /> */}
      {/* <EventRoleHeatmap data={data} /> */}
      {/* <EventRoleGraph data={data} /> */}
    </div>
  );
};

export default ChartTab;
