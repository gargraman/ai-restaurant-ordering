/**
 * API Client
 * Handles all communication with the backend restaurant catering API
 */

import type { SearchRequest, SearchResponse, SessionResponse, MenuItemResult, CartItem } from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

/**
 * Generate a unique session ID for the chat session
 */
export function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Execute a search query against the hybrid search backend
 */
export async function executeSearch(
  request: SearchRequest,
  signal?: AbortSignal
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/chat/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Search failed: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Get session details including tracked entities and conversation history
 */
export async function getSession(sessionId: string): Promise<SessionResponse> {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}`, {
    headers: {
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch session: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Send feedback on a search result
 */
export async function sendFeedback(
  sessionId: string,
  docId: string,
  rating: number,
  comment?: string
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}/feedback`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      doc_id: docId,
      rating,
      comment,
    }),
  });

  if (!response.ok) {
    throw new Error(`Feedback failed: ${response.statusText}`);
  }
}

/**
 * Convert a MenuItemResult to a CartItem
 */
export function menuItemToCartItem(
  item: MenuItemResult,
  quantity: number = 1
): CartItem {
  return {
    id: item.doc_id,
    name: item.item_name,
    description: item.item_description || undefined,
    price: item.display_price || item.price_per_person || 0,
    quantity,
    restaurantName: item.restaurant_name,
    image: undefined, // Images not included in MenuItemResult
  };
}
