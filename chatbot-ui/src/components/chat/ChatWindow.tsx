/**
 * ChatWindow Component
 * Main chat interface with message list, input, and rich interactions
 */

"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { toast } from "sonner";
import type { Message, MenuItemResult } from "@/types/api";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";
import { OfferCard } from "./OfferCard";
import {
  Send,
  Mic,
  StopCircle,
  Sparkles,
  UtensilsCrossed,
} from "lucide-react";

const QUICK_SUGGESTIONS = [
  "Italian catering for 20 people",
  "Vegetarian options under $15",
  "Best deals for corporate events",
  "Gluten-free menu suggestions",
];

const SAMPLE_OFFERS = [
  {
    title: "First Order Discount",
    description: "Get 20% off on your first catering order!",
    discount: "20% OFF",
    code: "FIRST20",
    validUntil: "Dec 31, 2025",
  },
];

interface ChatWindowProps {
  onSendMessage: (message: string) => Promise<void>;
  isLoading: boolean;
  messages: Message[];
  onAddToCart: (item: MenuItemResult) => void;
  onQuickReply: (reply: string) => void;
}

export function ChatWindow({
  onSendMessage,
  isLoading,
  messages,
  onAddToCart,
  onQuickReply,
}: ChatWindowProps) {
  const [inputValue, setInputValue] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!inputValue.trim() || isLoading) return;

    const message = inputValue.trim();
    setInputValue("");

    try {
      await onSendMessage(message);
    } catch {
      toast.error("Failed to send message. Please try again.");
      setInputValue(message); // Restore message on error
    }
  };

  const handleQuickReply = (reply: string) => {
    onQuickReply(reply);
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    toast.info(
      isRecording ? "Voice recording stopped" : "Listening... Speak now"
    );
    // In production, integrate with Web Speech API or similar
  };

  return (
    <div className="flex h-full flex-col">
      {/* Header — premium gradient */}
      <header className="bg-gradient-to-r from-orange-600 to-orange-500 px-5 py-4 shadow-lg">
        <div className="flex items-center gap-4">
          {/* Bot icon in white circle */}
          <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-full bg-white shadow-md">
            <UtensilsCrossed className="h-5 w-5 text-orange-600" />
          </div>

          <div className="flex-1">
            <h1 className="text-base font-bold leading-tight text-white">
              Restaurant Catering Bot
            </h1>
            {/* Status indicator */}
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-300 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-green-400" />
              </span>
              <span className="text-xs text-orange-100">Online</span>
            </div>
          </div>

          {/* AI Powered chip */}
          <div className="flex items-center gap-1.5 rounded-full border border-white/20 bg-white/10 px-3 py-1 backdrop-blur-sm">
            <Sparkles className="h-3 w-3 text-white" />
            <span className="text-xs font-medium text-white">AI Powered</span>
          </div>
        </div>
      </header>

      {/* Messages Area */}
      <ScrollArea className="flex-1 bg-gray-50/50">
        <div className="flex min-h-full flex-col justify-end p-4">
          {/* Welcome Screen */}
          {messages.length === 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mx-auto max-w-2xl space-y-6 py-12"
            >
              <div className="text-center">
                <h2 className="mb-3 text-3xl font-bold leading-tight">
                  <span className="bg-gradient-to-r from-orange-600 to-orange-500 bg-clip-text text-transparent">
                    Find Perfect Catering
                  </span>
                </h2>
                <p className="text-gray-500">
                  I&apos;m your AI catering assistant. I can help you find
                  perfect menu options for your events.
                </p>
              </div>

              {/* Quick Suggestion Cards */}
              <div className="grid gap-3 sm:grid-cols-2">
                {QUICK_SUGGESTIONS.map((suggestion, index) => (
                  <motion.button
                    key={suggestion}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    onClick={() => handleQuickReply(suggestion)}
                    className="group rounded-xl border border-gray-200 bg-white p-4 text-left text-sm text-gray-700 shadow-sm transition-all duration-200 hover:border-orange-400 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500"
                  >
                    <span className="block border-l-2 border-transparent pl-2 transition-colors duration-200 group-hover:border-orange-500">
                      {suggestion}
                    </span>
                  </motion.button>
                ))}
              </div>

              {/* Offer — centred, max-w-md */}
              <div className="mx-auto max-w-md">
                {SAMPLE_OFFERS.map((offer, index) => (
                  <OfferCard key={offer.title} {...offer} index={index} />
                ))}
              </div>
            </motion.div>
          )}

          {/* Messages */}
          <div className="flex flex-col gap-4">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onAddToCart={onAddToCart}
                onQuickReply={handleQuickReply}
              />
            ))}
          </div>

          {/* Typing Indicator */}
          {isLoading && <TypingIndicator />}

          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Gradient fade above input */}
      <div className="pointer-events-none h-8 bg-gradient-to-t from-white to-transparent" />

      {/* Input Area */}
      <div className="border-t border-gray-100 bg-white px-4 pb-4 pt-2">
        <form onSubmit={handleSubmit} className="mx-auto max-w-4xl">
          <Card className="flex items-center gap-2 p-2 shadow-sm ring-1 ring-gray-200 transition-shadow focus-within:ring-2 focus-within:ring-orange-400">
            <Input
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about catering options, menus, prices..."
              disabled={isLoading}
              maxLength={1000}
              className="flex-1 border-0 focus-visible:ring-0"
              aria-label="Chat message input"
            />

            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={toggleRecording}
              disabled={isLoading}
              className={`text-gray-400 transition-colors hover:bg-orange-50 hover:text-orange-600 ${isRecording ? "animate-pulse text-red-500" : ""}`}
              aria-label={isRecording ? "Stop recording" : "Start voice input"}
            >
              {isRecording ? (
                <StopCircle className="h-5 w-5" />
              ) : (
                <Mic className="h-5 w-5" />
              )}
            </Button>

            <Button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="bg-gradient-to-r from-orange-500 to-orange-600 text-white shadow-sm hover:from-orange-600 hover:to-orange-700 disabled:opacity-50"
              aria-label="Send message"
            >
              <Send className="h-5 w-5" />
            </Button>
          </Card>

          <p className="mt-2 text-center text-xs text-gray-400">
            AI-powered responses may vary. Verify important details with
            restaurants.
          </p>
        </form>
      </div>
    </div>
  );
}
