import React from 'react';
import { Tag, Tooltip } from 'antd';
import { Typography } from 'antd';

const { Title } = Typography;

interface EventTypeDisplayProps {
  eventType: string;
  showOriginal?: boolean;
  size?: 'small' | 'default' | 'large';
  color?: string;
}

const EventTypeDisplay: React.FC<EventTypeDisplayProps> = ({ 
  eventType, 
  showOriginal = false, 
  size = 'default',
  color 
}) => {
  // Mapping từ event type sang tên tiếng Việt
  const eventTypeMapping: Record<string, string> = {
    "ArtifactExistence.DamageDestroyDisableDismantle.Damage": "Hư hỏng tài sản",
    "ArtifactExistence.DamageDestroyDisableDismantle.Destroy": "Phá hủy tài sản",
    "ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse": "Vô hiệu hóa bom",
    "ArtifactExistence.DamageDestroyDisableDismantle.Dismantle": "Tháo dỡ tài sản",
    "ArtifactExistence.DamageDestroyDisableDismantle.Unspecified": "Hư hỏng tài sản",
    "ArtifactExistence.ManufactureAssemble.Unspecified": "Sản xuất/lắp ráp",
    
    "Cognitive.IdentifyCategorize.Unspecified": "Nhận dạng/phân loại",
    "Cognitive.Inspection.SensoryObserve": "Quan sát/kiểm tra",
    "Cognitive.Research.Unspecified": "Nghiên cứu",
    "Cognitive.TeachingTrainingLearning.Unspecified": "Giảng dạy/học tập",
    
    "Conflict.Attack.DetonateExplode": "Tấn công bằng bom",
    "Conflict.Attack.Unspecified": "Tấn công",
    "Conflict.Defeat.Unspecified": "Đánh bại",
    "Conflict.Demonstrate.DemonstrateWithViolence": "Biểu tình có bạo lực",
    "Conflict.Demonstrate.Unspecified": "Biểu tình",
    
    "Contact.Contact.Broadcast": "Phát sóng",
    "Contact.Contact.Correspondence": "Thư từ",
    "Contact.Contact.Meet": "Gặp gỡ",
    "Contact.Contact.Unspecified": "Liên lạc",
    "Contact.RequestCommand.Broadcast": "Yêu cầu qua phát sóng",
    "Contact.RequestCommand.Meet": "Yêu cầu qua gặp gỡ",
    "Contact.RequestCommand.Unspecified": "Yêu cầu/lệnh",
    "Contact.ThreatenCoerce.Broadcast": "Đe dọa qua phát sóng",
    "Contact.ThreatenCoerce.Correspondence": "Đe dọa qua thư từ",
    "Contact.ThreatenCoerce.Unspecified": "Đe dọa/cưỡng ép",
    
    "Control.ImpedeInterfereWith.Unspecified": "Cản trở/can thiệp",
    
    "Disaster.Crash.Unspecified": "Tai nạn/va chạm",
    "Disaster.DiseaseOutbreak.Unspecified": "Dịch bệnh bùng phát",
    
    "GenericCrime.GenericCrime.GenericCrime": "Tội phạm",
    
    "Justice.Acquit.Unspecified": "Tuyên bố vô tội",
    "Justice.ArrestJailDetain.Unspecified": "Bắt giữ",
    "Justice.ChargeIndict.Unspecified": "Buộc tội",
    "Justice.Convict.Unspecified": "Kết án có tội",
    "Justice.InvestigateCrime.Unspecified": "Điều tra tội phạm",
    "Justice.ReleaseParole.Unspecified": "Thả tự do",
    "Justice.Sentence.Unspecified": "Án phạt",
    "Justice.TrialHearing.Unspecified": "Xét xử",
    
    "Life.Die.Unspecified": "Cái chết",
    "Life.Infect.Unspecified": "Nhiễm bệnh",
    "Life.Injure.Unspecified": "Bị thương",
    
    "Medical.Intervention.Unspecified": "Can thiệp y tế",
    
    "Movement.Transportation.Evacuation": "Sơ tán",
    "Movement.Transportation.IllegalTransportation": "Vận chuyển bất hợp pháp",
    "Movement.Transportation.PreventPassage": "Ngăn cản di chuyển",
    "Movement.Transportation.Unspecified": "Vận chuyển",
    
    "Personnel.EndPosition.Unspecified": "Kết thúc vị trí",
    "Personnel.StartPosition.Unspecified": "Bắt đầu vị trí",
    
    "Transaction.Donation.Unspecified": "Quyên góp",
    "Transaction.ExchangeBuySell.Unspecified": "Mua bán/trao đổi"
  };

  // Lấy tên tiếng Việt
  const vietnameseName = eventTypeMapping[eventType] || eventType;
  
  // Lấy category (phần đầu của event type)
  const category = eventType.split('.')[0];
  
  // Màu sắc theo category
  const getCategoryColor = (category: string): string => {
    const colorMap: Record<string, string> = {
      'Conflict': 'red',
      'Life': 'volcano',
      'Justice': 'blue',
      'Contact': 'green',
      'Disaster': 'orange',
      'ArtifactExistence': 'purple',
      'Cognitive': 'cyan',
      'Control': 'magenta',
      'GenericCrime': 'red',
      'Medical': 'lime',
      'Movement': 'geekblue',
      'Personnel': 'gold',
      'Transaction': 'green'
    };
    return colorMap[category] || 'default';
  };

  const tagColor = color || getCategoryColor(category);

  return (
    <Tooltip 
      title={showOriginal ? vietnameseName : eventType}
      placement="top"
    >
      <Title level={4} style={{ margin: 0, display: 'inline' }}>Sự kiện: {vietnameseName}</Title>
      {/* <Tag 
        color={tagColor} 
        style={{ 
          margin: '2px',
          fontSize: size === 'small' ? '11px' : size === 'large' ? '14px' : '12px'
        }}
      >
        {vietnameseName}
      </Tag> */}
    </Tooltip>
  );
};

export default EventTypeDisplay;