"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

type LoadingVariant = "spinner" | "skeleton" | "dots";

interface LoadingStateProps {
  message?: string;
  variant?: LoadingVariant;
  className?: string;
}

// --- Spinner ---
function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg
      className={cn("animate-spin text-primary", className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

// --- Dots ---
function DotsLoader() {
  return (
    <div className="flex items-center gap-1.5" aria-hidden="true">
      {[0, 0.15, 0.3].map((delay, i) => (
        <motion.span
          key={i}
          className="h-2.5 w-2.5 rounded-full bg-primary"
          animate={{
            y: [0, -6, 0],
            opacity: [0.6, 1, 0.6],
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}

// --- Skeleton layout ---
function SkeletonLayout() {
  return (
    <div className="w-full space-y-3">
      <div className="flex items-center gap-3">
        <Skeleton className="h-10 w-10 rounded-full" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-3 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
        </div>
      </div>
      <Skeleton className="h-24 w-full rounded-lg" />
      <div className="space-y-2">
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-5/6" />
        <Skeleton className="h-3 w-4/6" />
      </div>
    </div>
  );
}

// --- Main component ---
function LoadingState({
  message,
  variant = "spinner",
  className,
}: LoadingStateProps) {
  if (variant === "skeleton") {
    return (
      <div
        className={cn("w-full", className)}
        role="status"
        aria-label={message ?? "Loading"}
      >
        <SkeletonLayout />
        {message && (
          <p className="sr-only">{message}</p>
        )}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 py-12",
        className
      )}
      role="status"
      aria-label={message ?? "Loading"}
    >
      {variant === "spinner" && <SpinnerIcon className="h-8 w-8" />}
      {variant === "dots" && <DotsLoader />}
      {message && (
        <p className="text-sm text-muted-foreground">{message}</p>
      )}
    </div>
  );
}

LoadingState.displayName = "LoadingState";

export { LoadingState, type LoadingStateProps, type LoadingVariant };
