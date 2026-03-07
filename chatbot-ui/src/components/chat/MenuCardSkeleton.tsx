/**
 * MenuCardSkeleton Component
 * Placeholder skeleton that mirrors the exact dimensions of MenuCard
 * while menu items are being loaded.
 */

import React from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface MenuCardSkeletonProps {
  /** Number of skeleton cards to render. Defaults to 3. */
  count?: number;
  className?: string;
}

/**
 * Single skeleton card that matches the MenuCard layout:
 * - Image area: h-40 (same as MenuCard's `h-40 w-full` block)
 * - Title line
 * - Two description lines
 * - Price + button row in a footer strip
 */
function SingleMenuCardSkeleton() {
  return (
    <div className="min-w-[280px] overflow-hidden rounded-xl border-0 bg-white shadow-lg">
      {/* Image block — matches `h-40 w-full` in MenuCard */}
      <Skeleton className="h-40 w-full rounded-none" />

      {/* Card content — matches CardContent p-4 */}
      <div className="space-y-3 p-4">
        {/* Item name — matches h3 text-lg font-semibold */}
        <Skeleton className="h-5 w-3/4" />

        {/* Description — 2 lines matching `line-clamp-2 text-sm` */}
        <div className="space-y-1.5">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-5/6" />
        </div>

        {/* Price — matches `text-xl font-bold` */}
        <Skeleton className="h-6 w-1/3" />
      </div>

      {/* Footer — matches CardFooter border-t bg-gray-50 p-3 */}
      <div className="border-t bg-gray-50 p-3">
        <Skeleton className="h-9 w-full rounded-md" />
      </div>
    </div>
  );
}

export function MenuCardSkeleton({
  count = 3,
  className,
}: MenuCardSkeletonProps) {
  return (
    <div
      className={cn("flex gap-3 overflow-x-auto pb-2", className)}
      aria-busy="true"
      aria-label="Loading menu items"
    >
      {Array.from({ length: count }, (_, i) => (
        <SingleMenuCardSkeleton key={i} />
      ))}
    </div>
  );
}

MenuCardSkeleton.displayName = "MenuCardSkeleton";
