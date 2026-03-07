"use client";

import React, { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ErrorPageProps {
  error: Error & { digest?: string };
  reset: () => void;
}

/**
 * Determines if a string looks like a stack trace or internal error detail
 * that should not be exposed to the user.
 */
function isSafeErrorMessage(message: string): boolean {
  // Reject messages that look like stack traces or file paths
  const unsafePatterns = [
    /at\s+\w+\s+\(/,           // "at FunctionName ("
    /^\s+at\s/m,               // indented "at " lines
    /\.(ts|tsx|js|jsx):\d+/,   // file.ts:42
    /Error:\s+.*\n/,           // multi-line Error: prefix
    /node_modules/,            // internal module paths
    /webpack|turbopack/i,      // bundler internals
  ];

  return !unsafePatterns.some((pattern) => pattern.test(message));
}

export default function ChatErrorPage({ error, reset }: ErrorPageProps) {
  const router = useRouter();

  useEffect(() => {
    console.error("[ChatErrorBoundary]", error);
  }, [error]);

  const displayMessage =
    error.message && isSafeErrorMessage(error.message)
      ? error.message
      : "We encountered an unexpected error.";

  return (
    <div
      className={cn(
        "flex min-h-screen flex-col items-center justify-center",
        "bg-gradient-to-br from-orange-50 to-white px-4"
      )}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="flex max-w-md flex-col items-center gap-6 text-center"
      >
        {/* Warning icon */}
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-red-50 ring-1 ring-red-100">
          <AlertTriangle
            className="h-10 w-10 text-red-500"
            strokeWidth={1.5}
            aria-hidden="true"
          />
        </div>

        {/* Title and description */}
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-gray-900">
            Something went wrong
          </h1>
          <p className="text-sm leading-relaxed text-gray-500">
            {displayMessage}
          </p>
          {error.digest && (
            <p className="text-xs text-gray-400">Error ID: {error.digest}</p>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex flex-col gap-3 sm:flex-row">
          <Button
            onClick={reset}
            className="bg-orange-600 hover:bg-orange-700 focus-visible:ring-orange-500"
          >
            Try again
          </Button>
          <Button
            variant="outline"
            onClick={() => router.back()}
            className="border-orange-200 text-orange-700 hover:bg-orange-50"
          >
            Go back
          </Button>
        </div>
      </motion.div>
    </div>
  );
}
