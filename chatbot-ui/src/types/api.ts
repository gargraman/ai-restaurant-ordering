/**
 * API Types for Restaurant Chatbot
 * Based on Hybrid Search v2 backend API
 */

export interface MenuItemResult {
  doc_id: string;
  restaurant_name: string;
  city: string;
  state: string;
  item_name: string;
  item_description?: string | null;
  display_price?: number | null;
  price_per_person?: number | null;
  serves_min?: number | null;
  serves_max?: number | null;
  dietary_labels: string[];
  tags: string[];
  rrf_score: number;
}

export interface SearchRequest {
  session_id: string;
  user_input: string;
  max_results?: number;
}

export interface SearchResponse {
  session_id: string;
  resolved_query: string;
  intent: string;
  is_follow_up: boolean;
  filters: Record<string, unknown>;
  results: MenuItemResult[];
  answer: string;
  confidence: number;
  processing_time_ms: number;
}

export interface SessionResponse {
  session_id: string;
  created_at: string;
  last_activity: string;
  entities: Record<string, unknown>;
  conversation_length: number;
  previous_results_count: number;
}

export interface FeedbackRequest {
  doc_id: string;
  rating: number;
  comment?: string | null;
}

export interface CartItem {
  id: string;
  name: string;
  description?: string;
  price: number;
  quantity: number;
  restaurantName: string;
  image?: string;
}

export interface OrderSummary {
  items: CartItem[];
  subtotal: number;
  tax: number;
  total: number;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  menuItems?: MenuItemResult[];
  type?: "text" | "menu" | "offer" | "order_confirmation" | "quick_replies";
  quickReplies?: string[];
}
