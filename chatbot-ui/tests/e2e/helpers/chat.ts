/**
 * Page object helpers for the chat interface.
 */

import type { Page, Locator } from "@playwright/test";

export class ChatPage {
  readonly page: Page;
  readonly messageInput: Locator;
  readonly sendButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.messageInput = page.getByLabel("Chat message input");
    this.sendButton = page.getByLabel("Send message");
  }

  async goto() {
    await this.page.goto("/chat");
    // Wait for the React hydration (initial greeting visible)
    await this.page.getByText(/restaurant catering assistant/i).waitFor({ timeout: 10000 });
  }

  /**
   * Fill and submit a message. Returns a promise so callers can race it with
   * waitForResponse if needed to avoid the race condition.
   */
  async sendMessage(text: string) {
    await this.messageInput.fill(text);
    await this.sendButton.click();
  }

  async sendMessageViaEnter(text: string) {
    await this.messageInput.fill(text);
    await this.messageInput.press("Enter");
  }
}
