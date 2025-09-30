import React from "react";
import { Card, Table, Tag } from "antd";
import type { SentenceResult } from "../types";

interface Props {
  data: SentenceResult[];
}

const columns = [
  {
    title: "Event Type",
    dataIndex: "event_type",
    key: "event_type",
    render: (text: string) => <Tag color="blue">{text}</Tag>,
  },
  {
    title: "Role",
    dataIndex: "role",
    key: "role",
    render: (text: string) => <Tag color="purple">{text}</Tag>,
  },
  {
    title: "Question",
    dataIndex: "question",
    key: "question",
    render: (text: string) => <span style={{ color: "#888" }}>{text}</span>,
  },
  {
    title: "Answer",
    dataIndex: "answer",
    key: "answer",
    render: (text: string) => <strong>{text}</strong>,
  },
];

const DetailTab: React.FC<Props> = ({ data }) => {
  return (
    <>
      {data.map((item, idx) => (
        <Card
          key={idx}
          title={
            <div style={{ fontWeight: 200, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {item.input}
            </div>
        }
          style={{ marginBottom: 24, boxShadow: "0 2px 8px #f0f1f2" }}
        >
          <Table
            dataSource={item.role_answers.map((r, i) => ({ ...r, key: i }))}
            columns={columns}
            pagination={false}
            size="middle"
            bordered
          />
        </Card>
      ))}
    </>
  );
};

export default DetailTab;