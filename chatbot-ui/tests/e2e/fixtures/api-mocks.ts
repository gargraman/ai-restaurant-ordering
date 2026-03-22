/**
 * Shared API mock data and route handlers for Playwright tests.
 * Intercepts requests to localhost:8000 so tests run without the backend.
 */

import type { Page, Route } from "@playwright/test";

export const MOCK_MENU_ITEMS = [
  {
    doc_id: "item-001",
    item_name: "Grilled Chicken Platter",
    item_description: "Tender grilled chicken with herbs and spices",
    restaurant_name: "Tasty Bites Catering",
    price_per_person: 18.5,
    display_price: 18.5,
    serves: 10,
    category: "Main Course",
    score: 0.92,
  },
  {
    doc_id: "item-002",
    item_name: "Vegetarian Pasta Buffet",
    item_description: "Assorted pasta dishes with fresh vegetables",
    restaurant_name: "Green Garden Eats",
    price_per_person: 14.0,
    display_price: 14.0,
    serves: 15,
    category: "Main Course",
    score: 0.87,
  },
];

export const MOCK_SEARCH_RESPONSE = {
  session_id: "session-test-123",
  answer:
    "I found some great catering options for you! Here are my top recommendations based on your request.",
  results: MOCK_MENU_ITEMS,
  intent: "search",
  processing_time_ms: 245,
  query_used: "chicken catering",
  follow_up_suggestions: [
    "Tell me more about the chicken platter",
    "Show me vegetarian options",
    "What's the minimum order?",
  ],
};

export const MOCK_EMPTY_SEARCH_RESPONSE = {
  session_id: "session-test-123",
  answer: "I couldn't find any items matching your request. Try a different search.",
  results: [],
  intent: "search",
  processing_time_ms: 120,
  query_used: "xyz unknown dish",
  follow_up_suggestions: [],
};

export const MOCK_HEALTH_RESPONSE = { status: "healthy" };

/**
 * Set up default API route mocks on a page.
 * Call this in beforeEach or at the start of a test.
 */
export async function setupApiMocks(page: Page) {
  // Health check
  await page.route("**/health", (route: Route) => {
    route.fulfill({ json: MOCK_HEALTH_RESPONSE });
  });

  // Search endpoint — default returns results
  await page.route("**/chat/search", (route: Route) => {
    route.fulfill({ json: MOCK_SEARCH_RESPONSE });
  });

  // Session endpoints
  await page.route("**/session/**", (route: Route) => {
    if (route.request().method() === "DELETE") {
      route.fulfill({ status: 200, body: "" });
    } else {
      route.fulfill({
        json: { session_id: "session-test-123", created_at: new Date().toISOString() },
      });
    }
  });
}

/**
 * Override the search route with a custom response for a specific test.
 */
export async function mockSearchResponse(page: Page, response: object, status = 200) {
  await page.route("**/chat/search", (route: Route) => {
    route.fulfill({ status, json: response });
  });
}

/**
 * Override search to simulate a network error.
 */
export async function mockSearchError(page: Page) {
  await page.route("**/chat/search", (route: Route) => {
    route.abort("failed");
  });
}
