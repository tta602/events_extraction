export interface RoleAnswer {
  event_type: string;
  role: string;
  question: string;
  answer: string;
}

export interface SentenceResult {
  input: string;
  index_input: number;
  role_answers: RoleAnswer[];
}

export interface RoleInfo {
  role: string;
  answer: string;
  sentence: string;
}

export interface EventSummary { 
  event_type: string; 
  frequency: number; 
  sentences: string[]; 
  total_roles: number;
  roles: RoleInfo[];  // Thông tin chi tiết về roles được phát hiện
} 
  
export interface SummaryResponse { 
  top_events: EventSummary[]; 
  total_sentences: number; 
  total_events: number; 
}