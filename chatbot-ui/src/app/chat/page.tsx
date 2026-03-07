/**
 * Chat Page
 * Main chat interface for restaurant catering bot
 */

"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "sonner";
import type { Message, MenuItemResult, CartItem, SearchResponse } from "@/types/api";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { OrderPanel } from "@/components/chat/OrderPanel";
import {
  executeSearch,
  generateSessionId,
  menuItemToCartItem,
} from "@/lib/api";
import { generateId } from "@/lib/utils";

export default function ChatPage() {
  // Session ID — initialized synchronously from localStorage to avoid a
  // render cycle where sessionId is "" and handleSendMessage bails early.
  const [sessionId] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    const stored = localStorage.getItem("chat_session_id");
    if (stored) return stored;
    const newId = generateSessionId();
    localStorage.setItem("chat_session_id", newId);
    return newId;
  });

  const [messages, setMessages] = useState<Message[]>(() => [
    {
      id: generateId(),
      role: "assistant",
      content: "Hi! I'm your restaurant catering assistant. How can I help you today?",
      timestamp: new Date(),
      type: "text",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isLoadingRef = useRef(false);

  // Cart — restored from localStorage so items survive page refresh.
  const [cartItems, setCartItems] = useState<CartItem[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const stored = localStorage.getItem("chat_cart_items");
      return stored ? (JSON.parse(stored) as CartItem[]) : [];
    } catch {
      return [];
    }
  });
  const [isOrderPanelCollapsed, setIsOrderPanelCollapsed] = useState(false);

  // Persist cart changes to localStorage
  useEffect(() => {
    localStorage.setItem("chat_cart_items", JSON.stringify(cartItems));
  }, [cartItems]);

  // Abort in-flight request on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  // Send message to API
  const handleSendMessage = useCallback(
    async (userInput: string) => {
      if (!sessionId || isLoadingRef.current) return;

      // Abort any previous in-flight request
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      isLoadingRef.current = true;
      setIsLoading(true);

      // Add user message
      const userMessage: Message = {
        id: generateId(),
        role: "user",
        content: userInput,
        timestamp: new Date(),
        type: "text",
      };

      setMessages((prev) => [...prev, userMessage]);

      try {
        // Call API
        const response: SearchResponse = await executeSearch(
          {
            session_id: sessionId,
            user_input: userInput,
            max_results: 10,
          },
          controller.signal
        );

        // Process response
        const assistantMessage: Message = {
          id: generateId(),
          role: "assistant",
          content: response.answer,
          timestamp: new Date(),
          type: response.results.length > 0 ? "menu" : "text",
          menuItems: response.results,
        };

        setMessages((prev) => [...prev, assistantMessage]);

        // Show toast with processing time
        toast.success(`Search completed in ${response.processing_time_ms}ms`, {
          description: `Found ${response.results.length} results`,
        });
      } catch (error) {
        // Ignore intentional aborts
        if (error instanceof Error && error.name === "AbortError") return;

        console.error("Search error:", error);

        // Add error message
        const errorMessage: Message = {
          id: generateId(),
          role: "assistant",
          content:
            "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
          timestamp: new Date(),
          type: "text",
        };

        setMessages((prev) => [...prev, errorMessage]);
        toast.error("Failed to get response. Please try again.");
      } finally {
        // Only clear loading if this is still the active request.
        // Prevents an aborted request's finally block from clobbering
        // the loading state of a newer in-flight request.
        if (abortControllerRef.current === controller) {
          isLoadingRef.current = false;
          setIsLoading(false);
        }
      }
    },
    [sessionId]
  );

  // Handle quick replies
  const handleQuickReply = useCallback(
    (reply: string) => {
      handleSendMessage(reply);
    },
    [handleSendMessage]
  );

  // Add item to cart
  const handleAddToCart = useCallback((item: MenuItemResult) => {
    const cartItem = menuItemToCartItem(item, 1);

    setCartItems((prev) => {
      const existing = prev.find((i) => i.id === item.doc_id);
      if (existing) {
        return prev.map((i) =>
          i.id === item.doc_id ? { ...i, quantity: i.quantity + 1 } : i
        );
      }
      return [...prev, cartItem];
    });

    toast.success(`${item.item_name} added to cart`, {
      description: `Price: $${item.display_price ?? item.price_per_person ?? 0}`,
    });

    // Expand order panel if collapsed
    setIsOrderPanelCollapsed(false);
  }, []);

  // Update cart item quantity
  const handleUpdateCartQuantity = useCallback(
    (itemId: string, quantity: number) => {
      if (quantity < 1) return;

      setCartItems((prev) =>
        prev.map((item) => (item.id === itemId ? { ...item, quantity } : item))
      );
    },
    []
  );

  // Remove item from cart
  const handleRemoveFromCart = useCallback((itemId: string) => {
    setCartItems((prev) => prev.filter((item) => item.id !== itemId));
    toast.success("Item removed from cart");
  }, []);

  // Handle checkout
  const handleCheckout = useCallback(() => {
    if (cartItems.length === 0) {
      toast.error("Your cart is empty");
      return;
    }

    // In production, integrate with backend order API
    toast.success("Checkout feature coming soon!", {
      description: "This will integrate with the order management system",
    });

    // Add order confirmation message
    const confirmationMessage: Message = {
      id: generateId(),
      role: "assistant",
      content: `Great choice! You've selected ${cartItems.length} items. 

**Order Summary:**
${cartItems.map((item) => `- ${item.name} x${item.quantity}`).join("\n")}

In production, this would proceed to payment and order confirmation.`,
      timestamp: new Date(),
      type: "order_confirmation",
    };

    setMessages((prev) => [...prev, confirmationMessage]);
  }, [cartItems]);

  // Toggle order panel
  const toggleOrderPanel = useCallback(() => {
    setIsOrderPanelCollapsed((prev) => !prev);
  }, []);

  return (
    <>
      <div className="flex h-screen overflow-hidden bg-gray-50">
        {/* Main Chat Area */}
        <main className="flex-1">
          <ChatWindow
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            messages={messages}
            onAddToCart={handleAddToCart}
            onQuickReply={handleQuickReply}
          />
        </main>

        {/* Order Panel — always mounted so its exit animation can play */}
        <OrderPanel
          items={cartItems}
          onUpdateQuantity={handleUpdateCartQuantity}
          onRemoveItem={handleRemoveFromCart}
          onCheckout={handleCheckout}
          isCollapsed={isOrderPanelCollapsed}
          onToggleCollapse={toggleOrderPanel}
        />
      </div>

      {/* Toast Notifications */}
      <Toaster
        richColors
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            borderRadius: "8px",
            background: "#fff",
            color: "#363636",
          },
        }}
      />
    </>
  );
}
