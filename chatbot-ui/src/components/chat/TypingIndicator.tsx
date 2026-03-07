/**
 * TypingIndicator Component
 * Shows animated typing indicator with bot avatar
 */

"use client";

import { motion } from "framer-motion";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { UtensilsCrossed } from "lucide-react";

export function TypingIndicator() {
  return (
    <div
      className="flex items-start gap-3 pt-2"
      role="status"
      aria-label="Assistant is typing"
    >
      {/* Icon-based avatar matching MessageBubble bot style */}
      <Avatar className="h-8 w-8 flex-shrink-0 ring-2 ring-orange-200">
        <AvatarFallback className="bg-gradient-to-br from-orange-500 to-orange-600 text-white">
          <UtensilsCrossed className="h-4 w-4" />
        </AvatarFallback>
      </Avatar>

      {/* Bubble matches assistant message style */}
      <div className="flex items-center gap-3 rounded-2xl rounded-tl-sm border border-gray-100 bg-white px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1">
          {[0, 0.2, 0.4].map((delay) => (
            <motion.div
              key={delay}
              className="h-2 w-2 rounded-full bg-orange-400"
              animate={{
                scale: [1, 1.3, 1],
                opacity: [0.4, 1, 0.4],
              }}
              transition={{
                duration: 0.7,
                repeat: Infinity,
                delay,
                ease: "easeInOut",
              }}
            />
          ))}
        </div>
        <span className="text-xs text-gray-400">Thinking...</span>
      </div>
    </div>
  );
}
