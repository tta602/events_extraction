import React from 'react';
import { Card, List, Statistic, Row, Col, Tag, Typography } from 'antd';
import type { SummaryResponse } from '../types';

const { Title, Text } = Typography;

interface SummaryTabProps {
  data: SummaryResponse;
}

const SummaryTab: React.FC<SummaryTabProps> = ({ data }) => {
  const { top_events, total_sentences, total_events } = data;

  return (
    <div>
      {/* Thống kê tổng quan */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="Tổng số câu"
              value={total_sentences}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Tổng số sự kiện"
              value={total_events}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Top sự kiện"
              value={top_events.length}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Danh sách top sự kiện */}
      <Card title="Top 3 Sự kiện Quan trọng Nhất" size="small">
        <List
          dataSource={top_events}
          renderItem={(event, index) => (
            <List.Item key={event.event_type}>
              <List.Item.Meta
                avatar={
                  <Tag color={index === 0 ? 'red' : index === 1 ? 'orange' : 'blue'}>
                    #{index + 1}
                  </Tag>
                }
                title={
                  <div>
                    <Title level={4} style={{ margin: 0, display: 'inline' }}>
                      {event.event_type}
                    </Title>
                    <Tag color="green" style={{ marginLeft: 8 }}>
                      {event.frequency} lần xuất hiện
                    </Tag>
                    <Tag color="purple">
                      {event.total_roles} roles
                    </Tag>
                  </div>
                }
                description={
                  <div>
                    <Text strong>Các câu chứa sự kiện:</Text>
                    <ul style={{ marginTop: 8, marginBottom: 0 }}>
                      {event.sentences.map((sentence, idx) => (
                        <li key={idx}>
                          <Text italic>"{sentence}"</Text>
                        </li>
                      ))}
                    </ul>
                  </div>
                }
              />
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default SummaryTab;