/**
 * FilterChips Component
 * Displays active filters with ability to clear individual filters or all
 */

"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";

interface FilterChipsProps {
  filters?: Record<string, unknown>;
  onFilterRemove?: (filterKey: string) => void;
  onClearAll?: () => void;
}

function formatFilterValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number") return value.toString();
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return String(value);
}

function getDisplayLabel(key: string, value: unknown): string | null {
  const displayValue = formatFilterValue(value);
  if (!displayValue) return null;

  const labels: Record<string, (val: string) => string> = {
    cuisine: (val) => `${val} cuisine`,
    dietary_labels: (val) => `${val}`,
    city: (val) => `${val}`,
    state: (val) => `${val}`,
    price_max: (val) => `Under $${val}`,
    price_per_person_max: (val) => `$${val} per person`,
    serves_min: (val) => `${val}+ people`,
    serves_max: (val) => `Up to ${val} people`,
    restaurant_name: (val) => `${val}`,
  };

  const formatter = labels[key];
  return formatter ? formatter(displayValue) : null;
}

export function FilterChips({
  filters = {},
  onFilterRemove,
  onClearAll,
}: FilterChipsProps) {
  // Filter out empty/null values and convert to array
  const activeFilters = Object.entries(filters)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => ({
      key,
      value,
      display: getDisplayLabel(key, value),
    }))
    .filter((f) => f.display !== null);

  if (activeFilters.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs font-medium text-gray-600">Filters:</span>
      <AnimatePresence mode="popLayout">
        {activeFilters.map(({ key, display }, index) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, scale: 0.75, x: -15 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            exit={{ opacity: 0, scale: 0.75, x: 15 }}
            transition={{
              type: "spring",
              stiffness: 250,
              damping: 20,
              delay: index * 0.05,
            }}
            className="flex items-center gap-1.5 rounded-full bg-gradient-to-r from-orange-50 to-orange-100 px-3 py-1.5 text-xs font-medium text-orange-700 border border-orange-200 shadow-sm hover:shadow-md transition-shadow"
          >
            <span>{display}</span>
            <motion.button
              onClick={() => onFilterRemove?.(key)}
              className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full hover:bg-orange-200 transition-colors"
              aria-label={`Remove ${display} filter`}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <X className="h-3 w-3" />
            </motion.button>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Clear all button */}
      {activeFilters.length > 0 && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClearAll}
          className="text-xs text-gray-500 hover:text-orange-600 underline font-medium"
        >
          Clear all
        </motion.button>
      )}
    </div>
  );
}
