import React, { useState } from "react";
import { Layout, Tabs, Spin, message } from "antd";
import type { TabsProps } from "antd";
import axios from "axios";
import type { SentenceResult, SummaryResponse } from "./types";
import DetailTab from "./components/DetailTab";
import ChartTab from "./components/ChartTab";
import SummaryTab from "./components/SummaryTab";
import TextInputPanel from "./components/TextInputPanel";


const { Sider, Content } = Layout;

const siderStyle: React.CSSProperties = {
  overflow: 'auto',
  background: "#fff",
  height: '100vh',
  position: 'sticky',
  insetInlineStart: 0,
  top: 0,
  bottom: 0,
  scrollbarWidth: 'thin',
  scrollbarGutter: 'stable',
};

const App: React.FC = () => {
  const [text, setText] = useState("");
  const [data, setData] = useState<SentenceResult[]>([]);
  const [summaryData, setSummaryData] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const handleExtract = async () => {
    if (!text.trim()) {
      message.warning("Bạn cần nhập văn bản trước!");
      return;
    }
    setLoading(true);
    try {
      // Gọi cả 2 API song song
      const [detailRes, summaryRes] = await Promise.all([
        axios.post<SentenceResult[]>("http://127.0.0.1:8000/extract", { text }),
        axios.post<SummaryResponse>("http://127.0.0.1:8000/extract-summary", { text })
      ]);
      
      // Xử lý kết quả chi tiết
      const filtered = detailRes.data.map((d) => ({
        ...d,
        role_answers: d.role_answers.filter((r) => !r.answer.includes("none")),
      }));
      setData(filtered);
      
      // Xử lý kết quả tổng hợp
      setSummaryData(summaryRes.data);
    } catch (err) {
      console.error(err);
      message.error("Gọi API thất bại!");
    } finally {
      setLoading(false);
    }
  };

  const items: TabsProps["items"] = [
    { key: "1", label: "Tổng hợp", children: summaryData ? <SummaryTab data={summaryData} /> : <div>Chưa có dữ liệu</div> },
    { key: "2", label: "Chi tiết", children: <DetailTab data={data} /> },
    { key: "3", label: "Biểu đồ", children: <ChartTab data={data} /> },
  ];

  return (
    <Layout hasSider>
      <Sider width="25%" style={siderStyle}>
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
            <Tabs 
              defaultActiveKey="1"
              items={items}
              tabBarStyle={{
                position: "sticky",
                top: 0,
                zIndex: 10,
                background: "#fff",
              }}
            />
          ) : (
            <div>Hãy nhập văn bản và bấm Extract để bắt đầu.</div>
          )}
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;