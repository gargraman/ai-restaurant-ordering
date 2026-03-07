import * as React from "react";
import { cn } from "@/lib/utils";

type TooltipSide = "top" | "bottom" | "left" | "right";

interface TooltipProps {
  content: string;
  children: React.ReactNode;
  side?: TooltipSide;
  className?: string;
}

const sideStyles: Record<TooltipSide, { tooltip: string; arrow: string }> = {
  top: {
    tooltip:
      "bottom-full left-1/2 mb-2 -translate-x-1/2",
    arrow:
      "top-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-b-transparent border-t-foreground/90",
  },
  bottom: {
    tooltip:
      "top-full left-1/2 mt-2 -translate-x-1/2",
    arrow:
      "bottom-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-t-transparent border-b-foreground/90",
  },
  left: {
    tooltip:
      "right-full top-1/2 mr-2 -translate-y-1/2",
    arrow:
      "left-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-r-transparent border-l-foreground/90",
  },
  right: {
    tooltip:
      "left-full top-1/2 ml-2 -translate-y-1/2",
    arrow:
      "right-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-l-transparent border-r-foreground/90",
  },
};

function Tooltip({ content, children, side = "top", className }: TooltipProps) {
  return (
    <span className={cn("group relative inline-flex", className)}>
      {children}

      {/* Tooltip bubble */}
      <span
        role="tooltip"
        className={cn(
          // layout & positioning
          "pointer-events-none absolute z-50 whitespace-nowrap",
          sideStyles[side].tooltip,
          // visual
          "rounded-md bg-foreground/90 px-2.5 py-1.5 text-xs font-medium text-background shadow-md",
          // visibility — hidden by default, visible on group hover/focus-within
          "opacity-0 transition-opacity duration-150",
          "group-hover:opacity-100 group-focus-within:opacity-100"
        )}
      >
        {content}

        {/* Arrow */}
        <span
          aria-hidden="true"
          className={cn(
            "pointer-events-none absolute h-0 w-0 border-4",
            sideStyles[side].arrow
          )}
        />
      </span>
    </span>
  );
}

Tooltip.displayName = "Tooltip";

export { Tooltip, type TooltipProps, type TooltipSide };
