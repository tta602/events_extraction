import React, { useEffect, useRef } from "react";
import * as G6 from "@antv/g6";
import { DagreLayout } from "@antv/layout";

interface Props {
  data: { role: string; answer: string }[];
}

const EventRoleGraph: React.FC<Props> = ({ data }) => {
  const ref = useRef<HTMLDivElement>(null);
  const graphRef = useRef<G6.Graph>();

  useEffect(() => {
    if (!ref.current) return;

    const filtered = data.filter((d) => d.answer && d.answer !== "None");

    const nodes: any[] = [];
    const edges: any[] = [];

    filtered.forEach((item) => {
      const roleId = `role-${item.role}`;
      const ansId = `ans-${item.answer}`;

      if (!nodes.find((n) => n.id === roleId)) {
        nodes.push({ id: roleId, label: item.role, type: "rect" });
      }
      if (!nodes.find((n) => n.id === ansId)) {
        nodes.push({ id: ansId, label: item.answer, type: "circle" });
      }

      edges.push({ source: roleId, target: ansId });
    });

    if (!graphRef.current) {
      graphRef.current = new G6.Graph({
        container: ref.current,
        width: 800,
        height: 500,
        layout: new DagreLayout({ rankdir: "LR" }),
        defaultNode: {
          size: [120, 40],
          style: { fill: "#E6F7FF", stroke: "#1890FF" },
          labelCfg: { style: { fontSize: 12 } },
        },
        defaultEdge: {
          style: { stroke: "#B5B5B5", lineWidth: 1 },
        },
        modes: { default: ["drag-canvas", "zoom-canvas", "drag-node"] },
      });
    }

    graphRef.current.data({ nodes, edges });
    graphRef.current.render();
    graphRef.current.fitView();

    return () => {
      graphRef.current?.destroy();
      graphRef.current = undefined;
    };
  }, [data]);

  return <div ref={ref} style={{ border: "1px solid #ddd" }} />;
};

export default EventRoleGraph;