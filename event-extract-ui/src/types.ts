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
