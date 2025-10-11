import React from 'react';
import { Card, List, Statistic, Row, Col, Tag, Typography, Divider } from 'antd';
import type { SummaryResponse } from '../types';
import EventTypeDisplay from './EventTypeDisplay';
import HighlightedText from './HighlightedText';

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
                  <div style={{ marginBottom: 16 }}>
                    <EventTypeDisplay 
                      eventType={event.event_type} 
                      size="large"
                      showOriginal={true}
                    />
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
                    {/* Hiển thị roles được phát hiện */}
                    {event.roles && event.roles.length > 0 && (
                      <div style={{ marginBottom: 16 }}>
                        <Text strong>🎯 Roles được phát hiện:</Text>
                        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                          {event.roles.map((roleInfo, idx) => (
                            <Tag key={idx} color="blue" style={{ marginBottom: 4 }}>
                              <Text strong>{roleInfo.role}:</Text> {roleInfo.answer}
                            </Tag>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <Divider style={{ margin: '12px 0' }} />
                    
                    {/* Hiển thị các câu với highlights */}
                    <Text strong>📝 Các câu chứa sự kiện:</Text>
                    <ul style={{ marginTop: 8, marginBottom: 0 }}>
                      {event.sentences.map((sentence, idx) => {
                        // Lấy tất cả answers từ roles của event này để highlight
                        const highlights = event.roles
                          .filter(r => r.sentence === sentence)
                          .map(r => r.answer);
                        
                        return (
                          <li key={idx} style={{ marginBottom: 8 }}>
                            <HighlightedText 
                              text={sentence} 
                              highlights={highlights}
                            />
                          </li>
                        );
                      })}
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