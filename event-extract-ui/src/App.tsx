import React, { useState } from "react";
import { Layout, Tabs, Spin, message } from "antd";
import type { TabsProps } from "antd";
import axios from "axios";
import type { SentenceResult } from "./types";
import DetailTab from "./components/DetailTab";
import ChartTab from "./components/ChartTab";
import TextInputPanel from "./components/TextInputPanel";


const { Sider, Content } = Layout;

const App: React.FC = () => {
  const [text, setText] = useState("");
  const [data, setData] = useState<SentenceResult[]>([]);
  const [loading, setLoading] = useState(false);

  const handleExtract = async () => {
    if (!text.trim()) {
      message.warning("Bạn cần nhập văn bản trước!");
      return;
    }
    setLoading(true);
    try {
      const res = await axios.post<SentenceResult[]>(
        "http://127.0.0.1:8000/extract",
        { text }
      );
      // bỏ answer = none
      const filtered = res.data.map((d) => ({
        ...d,
        role_answers: d.role_answers.filter((r) => !r.answer.includes("none")),
      }));
      setData(filtered);
    } catch (err) {
      console.error(err);
      message.error("Gọi API thất bại!");
    } finally {
      setLoading(false);
    }
  };

  const items: TabsProps["items"] = [
    { key: "1", label: "Chi tiết", children: <DetailTab data={data} /> },
    { key: "2", label: "Biểu đồ", children: <ChartTab data={data} /> },
  ];

  return (
    <Layout style={{ height: "100vh" }}>
      <Sider width="25%" style={{ background: "#fff" }}>
        <TextInputPanel
          text={text}
          setText={setText}
          onExtract={handleExtract}
          loading={loading}
        />
      </Sider>
      <Layout>
        <Content style={{ padding: "16px", background: "#fff" }}>
          {loading ? (
            <Spin size="large" />
          ) : data.length > 0 ? (
            <Tabs defaultActiveKey="1" items={items} />
          ) : (
            <div>Hãy nhập văn bản và bấm Extract để bắt đầu.</div>
          )}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;
