import React from 'react';
import { Typography } from 'antd';

const { Text } = Typography;

interface HighlightedTextProps {
  text: string;
  highlights: string[];  // Danh sách các từ/cụm từ cần highlight
}

const HighlightedText: React.FC<HighlightedTextProps> = ({ text, highlights }) => {
  if (!highlights || highlights.length === 0) {
    return <Text>{text}</Text>;
  }

  // Tạo regex pattern để match tất cả highlights (case insensitive)
  // Escape special regex characters và filter empty strings
  const validHighlights = highlights
    .filter(h => h && h.trim() !== '' && h.toLowerCase() !== 'none')
    .map(h => h.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  
  if (validHighlights.length === 0) {
    return <Text>{text}</Text>;
  }

  const pattern = validHighlights.join('|');
  const regex = new RegExp(`(${pattern})`, 'gi');
  
  // Split text by matches
  const parts = text.split(regex);
  
  return (
    <Text>
      {parts.map((part, index) => {
        // Check if this part matches any highlight (case insensitive)
        const isHighlight = validHighlights.some(
          h => part.toLowerCase() === h.toLowerCase()
        );
        
        if (isHighlight) {
          return (
            <Text 
              key={index} 
              strong 
              style={{ 
                backgroundColor: '#fff3cd', 
                padding: '0 4px',
                borderRadius: '2px',
                color: '#856404'
              }}
            >
              {part}
            </Text>
          );
        }
        return <span key={index}>{part}</span>;
      })}
    </Text>
  );
};

export default HighlightedText;