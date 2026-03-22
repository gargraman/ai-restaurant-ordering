/**
 * E2E tests for the Chat interface.
 * API calls to localhost:8000 are intercepted so no backend is needed.
 */

import { test, expect } from "@playwright/test";
import {
  setupApiMocks,
  mockSearchResponse,
  mockSearchError,
  MOCK_SEARCH_RESPONSE,
  MOCK_EMPTY_SEARCH_RESPONSE,
} from "./fixtures/api-mocks";
import { ChatPage } from "./helpers/chat";

test.describe("Chat Page", () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage once before each test (addInitScript runs before page load)
    await page.addInitScript(() => localStorage.clear());
    await setupApiMocks(page);
  });

  // ─── Page Load ────────────────────────────────────────────────────────────

  test("loads and redirects / to /chat", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/chat/);
  });

  test("shows the initial greeting from the assistant", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await expect(
      page.getByText(/restaurant catering assistant/i).first()
    ).toBeVisible();
  });

  test("renders the chat input and send button", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await expect(chat.messageInput).toBeVisible();
    await expect(chat.sendButton).toBeVisible();
  });

  test("send button is disabled when input is empty", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await expect(chat.sendButton).toBeDisabled();
  });

  test("renders the order panel", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    // The order panel / cart sidebar should be in the DOM
    const panel = page.locator("aside, [class*='order'], [class*='cart']").first();
    await expect(panel).toBeAttached();
  });

  // ─── Sending Messages ─────────────────────────────────────────────────────

  test("user can type and send a message", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    const userQuery = "Show me chicken catering options";
    await chat.messageInput.fill(userQuery);
    await expect(chat.messageInput).toHaveValue(userQuery);

    // sendButton enabled after typing
    await expect(chat.sendButton).toBeEnabled();
    await chat.sendButton.click();

    // User message appears in the chat
    await expect(page.getByText(userQuery)).toBeVisible();
  });

  test("send button is disabled while loading", async ({ page }) => {
    // Delay the API response so we can observe the loading state
    await page.route("**/chat/search", async (route) => {
      await new Promise((r) => setTimeout(r, 800));
      route.fulfill({ json: MOCK_SEARCH_RESPONSE });
    });

    const chat = new ChatPage(page);
    await chat.goto();

    await chat.messageInput.fill("chicken options");
    await chat.sendButton.click();

    await expect(chat.sendButton).toBeDisabled();
    await page.waitForResponse("**/chat/search");
  });

  test("pressing Enter sends the message", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    const [response] = await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessageViaEnter("vegetarian options please"),
    ]);

    expect(response.status()).toBe(200);
    await expect(page.getByText("vegetarian options please")).toBeVisible();
  });

  test("input is cleared after sending a message", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("any available items?"),
    ]);

    await expect(chat.messageInput).toHaveValue("");
  });

  // ─── API Response Handling ────────────────────────────────────────────────

  test("displays assistant response after a search", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("chicken catering"),
    ]);

    await expect(
      page.getByText(MOCK_SEARCH_RESPONSE.answer)
    ).toBeVisible({ timeout: 8000 });
  });

  test("renders menu cards when results are returned", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("chicken catering"),
    ]);

    await expect(
      page.getByText(MOCK_SEARCH_RESPONSE.results[0].item_name)
    ).toBeVisible({ timeout: 8000 });
    await expect(
      page.getByText(MOCK_SEARCH_RESPONSE.results[1].item_name)
    ).toBeVisible({ timeout: 8000 });
  });

  test("shows no-results message when search returns empty", async ({ page }) => {
    await mockSearchResponse(page, MOCK_EMPTY_SEARCH_RESPONSE);

    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("xyz unknown dish 12345"),
    ]);

    await expect(
      page.getByText(/couldn't find|no items|try a different/i)
    ).toBeVisible({ timeout: 8000 });
  });

  test("shows error message when API call fails", async ({ page }) => {
    await mockSearchError(page);

    const chat = new ChatPage(page);
    await chat.goto();

    // Fire and forget — the request will abort, so no response to wait on
    await chat.sendMessage("trigger error");

    // Match the message bubble specifically (not the toast which also appears)
    await expect(
      page.getByText(/i apologize|trouble processing|having trouble/i).first()
    ).toBeVisible({ timeout: 8000 });
  });

  // ─── Cart / Order Panel ───────────────────────────────────────────────────

  test("adds item to cart and shows in order panel", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("chicken catering"),
    ]);

    // Wait for menu cards to appear
    const itemName = MOCK_SEARCH_RESPONSE.results[0].item_name;
    const addButton = page.getByLabel(`Add ${itemName} to cart`);
    await expect(addButton).toBeVisible({ timeout: 8000 });
    await addButton.click();

    // Item should now appear in the order panel
    const orderPanel = page.getByLabel("Order summary");
    await expect(orderPanel.getByText(itemName)).toBeVisible({ timeout: 5000 });
  });

  test("shows toast notification when item added to cart", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("chicken catering"),
    ]);

    const itemName = MOCK_SEARCH_RESPONSE.results[0].item_name;
    const addButton = page.getByLabel(`Add ${itemName} to cart`);
    await expect(addButton).toBeVisible({ timeout: 8000 });
    await addButton.click();

    // Sonner toast notification
    await expect(page.getByText(/added to cart/i)).toBeVisible({ timeout: 5000 });
  });

  // ─── Session Persistence ──────────────────────────────────────────────────

  test("session ID is stored in localStorage after page load", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    // Poll until React sets the key (hydration runs after load)
    const sessionId = await page.waitForFunction(() =>
      localStorage.getItem("chat_session_id")
    );
    const value = await sessionId.jsonValue();
    expect(value).toBeTruthy();
    expect(String(value)).toMatch(/^session_/);
  });

  test("cart items are persisted in localStorage", async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("chicken catering"),
    ]);

    const itemName = MOCK_SEARCH_RESPONSE.results[0].item_name;
    const addButton = page.getByLabel(`Add ${itemName} to cart`);
    await expect(addButton).toBeVisible({ timeout: 8000 });
    await addButton.click();

    // Give React a tick to flush the useEffect that persists to localStorage
    await page.waitForFunction(() => {
      const data = localStorage.getItem("chat_cart_items");
      if (!data) return false;
      const cart = JSON.parse(data);
      return Array.isArray(cart) && cart.length > 0;
    });

    const cartData = await page.evaluate(() =>
      localStorage.getItem("chat_cart_items")
    );
    const cart = JSON.parse(cartData!);
    expect(cart.length).toBeGreaterThan(0);
  });

  // ─── API Request Validation ───────────────────────────────────────────────

  test("sends correct payload to /chat/search", async ({ page }) => {
    let capturedBody: Record<string, unknown> | null = null;

    await page.route("**/chat/search", async (route) => {
      capturedBody = JSON.parse(route.request().postData() ?? "{}");
      route.fulfill({ json: MOCK_SEARCH_RESPONSE });
    });

    const chat = new ChatPage(page);
    await chat.goto();

    await Promise.all([
      page.waitForResponse("**/chat/search"),
      chat.sendMessage("vegan buffet options"),
    ]);

    expect(capturedBody).not.toBeNull();
    expect(capturedBody!.user_input).toBe("vegan buffet options");
    expect(capturedBody!.session_id).toBeTruthy();
    expect(capturedBody!.max_results).toBeDefined();
  });
});
