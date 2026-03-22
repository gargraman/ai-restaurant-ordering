/**
 * SuggestionChips Component
 * Displays context-aware follow-up suggestions based on intent and conversation state
 */

"use client";

import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import {
  Filter,
  Utensils,
  TrendingUp,
  Share2,
  Clock,
  Users,
} from "lucide-react";

interface SuggestionChipsProps {
  intent?: string;
  resultsCount?: number;
  onSuggestClick: (suggestion: string) => void;
}

function getSuggestionsByIntent(
  intent: string | undefined,
  resultsCount: number = 0
): { text: string; icon: React.ReactNode }[] {
  const baseIcon = <Utensils className="h-3.5 w-3.5" />;

  switch (intent?.toLowerCase()) {
    case "search":
      return [
        {
          text: "Filter by dietary needs",
          icon: <Filter className="h-3.5 w-3.5" />,
        },
        { text: "Show similar items", icon: baseIcon },
        {
          text: "Refine by price range",
          icon: <TrendingUp className="h-3.5 w-3.5" />,
        },
      ];
    case "filter":
      return [
        { text: "Show more results", icon: baseIcon },
        { text: "Compare restaurants", icon: <Share2 className="h-3.5 w-3.5" /> },
        { text: "View popular items", icon: <TrendingUp className="h-3.5 w-3.5" /> },
      ];
    case "compare":
      return [
        { text: "See side-by-side details", icon: baseIcon },
        { text: "Check availability", icon: <Clock className="h-3.5 w-3.5" /> },
        {
          text: "Find catering packages",
          icon: <Users className="h-3.5 w-3.5" />,
        },
      ];
    case "clarify":
      return [
        {
          text: "Browse all restaurants",
          icon: <Utensils className="h-3.5 w-3.5" />,
        },
        { text: "View trending items", icon: <TrendingUp className="h-3.5 w-3.5" /> },
        {
          text: "Get personalized recommendations",
          icon: <Filter className="h-3.5 w-3.5" />,
        },
      ];
    default:
      return [
        { text: "Search by cuisine", icon: baseIcon },
        {
          text: "Filter by dietary preferences",
          icon: <Filter className="h-3.5 w-3.5" />,
        },
        {
          text: "Find budget-friendly options",
          icon: <TrendingUp className="h-3.5 w-3.5" />,
        },
      ];
  }
}

export function SuggestionChips({
  intent,
  resultsCount = 0,
  onSuggestClick,
}: SuggestionChipsProps) {
  const suggestions = getSuggestionsByIntent(intent, resultsCount);

  return (
    <div className="flex flex-wrap gap-2">
      {suggestions.map((suggestion, index) => (
        <motion.div
          key={suggestion.text}
          initial={{ opacity: 0, scale: 0.85, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{
            delay: index * 0.08,
            duration: 0.3,
            type: "spring",
            stiffness: 200,
            damping: 20,
          }}
          whileHover={{
            scale: 1.08,
            transition: { duration: 0.2 },
          }}
          whileTap={{ scale: 0.95 }}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => onSuggestClick(suggestion.text)}
            className="flex items-center gap-1.5 rounded-full border-orange-200 bg-orange-50 text-xs font-medium text-orange-700 transition-all duration-200 hover:bg-orange-100 hover:text-orange-800 hover:shadow-md active:shadow-sm"
          >
            {suggestion.icon}
            {suggestion.text}
          </Button>
        </motion.div>
      ))}
    </div>
  );
}
