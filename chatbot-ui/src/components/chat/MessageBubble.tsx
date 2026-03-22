/**
 * MessageBubble Component
 * Displays individual chat messages with avatar, timestamp, and rich content
 */

"use client";

import React from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { formatTime } from "@/lib/utils";
import type { Message, MenuItemResult } from "@/types/api";
import { MenuCard } from "./MenuCard";
import { MenuCardSkeleton } from "./MenuCardSkeleton";
import { UtensilsCrossed, User, SearchIcon, FilterIcon, HelpCircle, BarChart3 } from "lucide-react";

interface MessageBubbleProps {
  message: Message;
  onAddToCart?: (item: MenuItemResult) => void;
  onQuickReply?: (reply: string) => void;
}

function getIntentIcon(intent: string) {
  switch (intent?.toLowerCase()) {
    case "search":
      return <SearchIcon className="h-3.5 w-3.5" />;
    case "filter":
      return <FilterIcon className="h-3.5 w-3.5" />;
    case "compare":
      return <BarChart3 className="h-3.5 w-3.5" />;
    case "clarify":
      return <HelpCircle className="h-3.5 w-3.5" />;
    default:
      return null;
  }
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.9) return "bg-green-50 text-green-700 border-green-200";
  if (confidence >= 0.7) return "bg-yellow-50 text-yellow-700 border-yellow-200";
  return "bg-orange-50 text-orange-700 border-orange-200";
}

export function MessageBubble({
  message,
  onAddToCart,
  onQuickReply,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const hasIntentBadge = message.intent && message.confidence !== undefined && !isUser;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      {/* Avatar */}
      <Avatar
        className={`h-8 w-8 flex-shrink-0 ring-2 ${
          isUser ? "ring-blue-200" : "ring-orange-200"
        }`}
      >
        {isUser ? (
          <AvatarFallback className="bg-gradient-to-br from-blue-500 to-blue-600 text-white">
            <User className="h-4 w-4" />
          </AvatarFallback>
        ) : (
          <AvatarFallback className="bg-gradient-to-br from-orange-500 to-orange-600 text-white">
            <UtensilsCrossed className="h-4 w-4" />
          </AvatarFallback>
        )}
      </Avatar>

      {/* Message Content */}
      <div
        className={`flex max-w-[75%] flex-col gap-1.5 ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        {/* Intent & Confidence Badge */}
        {hasIntentBadge && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: -5 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{
              duration: 0.3,
              type: "spring",
              stiffness: 200,
              damping: 20,
            }}
            className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium border shadow-sm ${getConfidenceColor(
              message.confidence!
            )}`}
          >
            <motion.span
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{
                duration: 2,
                repeat: Infinity,
                repeatDelay: 3,
              }}
            >
              {getIntentIcon(message.intent)}
            </motion.span>
            <span className="capitalize">{message.intent}</span>
            <span className="text-xs opacity-75">
              {Math.round(message.confidence! * 100)}%
            </span>
          </motion.div>
        )}

        <div
          className={`px-4 py-3 ${
            isUser
              ? "rounded-2xl rounded-tr-sm bg-gradient-to-br from-orange-500 to-orange-600 text-white shadow-md"
              : "rounded-2xl rounded-tl-sm border border-gray-100 bg-white text-gray-900 shadow-sm"
          }`}
        >
          {/* Text Content */}
          {message.content && (
            <div className={`prose prose-sm max-w-none ${isUser ? "prose-invert" : ""}`}>
              <ReactMarkdown
                rehypePlugins={[rehypeSanitize]}
                components={{
                  p: ({ children }) => (
                    <p className="mb-2 last:mb-0">{children}</p>
                  ),
                  strong: ({ children }) => (
                    <strong className="font-semibold">{children}</strong>
                  ),
                  ul: ({ children }) => (
                    <ul className="ml-4 list-disc">{children}</ul>
                  ),
                  li: ({ children }) => (
                    <li className="my-1">{children}</li>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}

          {/* Menu Items Carousel — skeleton while loading, cards once populated */}
          {message.type === "menu" && message.menuItems?.length === 0 && (
            <div className="mt-3">
              <MenuCardSkeleton count={3} />
            </div>
          )}

          {message.menuItems && message.menuItems.length > 0 && (
            <div className="mt-3">
              <p className="mb-2 text-xs font-medium text-gray-500">
                Showing {message.menuItems.length} result
                {message.menuItems.length !== 1 ? "s" : ""}
              </p>
              <div
                className="flex gap-3 overflow-x-auto pb-2"
                style={{ scrollSnapType: "x mandatory" }}
              >
                {message.menuItems.map((item, index) => (
                  <div
                    key={item.doc_id}
                    className="min-w-[280px]"
                    style={{ scrollSnapAlign: "start" }}
                  >
                    <MenuCard
                      item={item}
                      onAddToCart={() => onAddToCart?.(item)}
                      index={index}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quick Reply Buttons */}
          {message.quickReplies && message.quickReplies.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {message.quickReplies.map((reply, index) => (
                <motion.button
                  key={reply}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => onQuickReply?.(reply)}
                  className={`rounded-full border px-4 py-1.5 text-sm font-medium transition-all duration-200 ${
                    isUser
                      ? "border-white/30 text-white hover:bg-white/20"
                      : "border-orange-400 text-orange-600 hover:bg-orange-500 hover:text-white"
                  }`}
                >
                  {reply}
                </motion.button>
              ))}
            </div>
          )}
        </div>

        {/* Timestamp */}
        <span className="text-[11px] text-gray-400">
          {formatTime(message.timestamp)}
        </span>
      </div>
    </motion.div>
  );
}
