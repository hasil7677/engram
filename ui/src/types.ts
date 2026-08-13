export interface ScoredMemory {
  memory_id: string;
  text: string;
  semantic_score: number;
  temporal_score: number;
  frequency_score: number;
  final_score: number;
  timestamp: string;
  source: string;
}

export interface EngramConfig {
  baseUrl: string;
  apiKey: string;
  userId: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
  error?: boolean;
}
