import React from "react";
import { Typography, Input, Button } from "antd";

const { Title } = Typography;
const { TextArea } = Input;

interface Props {
  text: string;
  setText: (t: string) => void;
  onExtract: () => void;
  loading: boolean;
}

const TextInputPanel: React.FC<Props> = ({ text, setText, onExtract, loading }) => {
  return (
    <div style={{ padding: "16px" }}>
      <Title level={4}>Nhập văn bản</Title>
      <TextArea
        rows={20}
        placeholder="Nhập văn bản ở đây..."
        style={{ resize: "none", marginBottom: 16 }}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <Button type="primary" block onClick={onExtract} loading={loading}>
        Extract Events
      </Button>
    </div>
  );
};

export default TextInputPanel;
