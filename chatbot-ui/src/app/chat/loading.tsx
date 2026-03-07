import React from "react";
import { LoadingState } from "@/components/ui/loading-state";

/**
 * Route-level loading UI for the /chat segment.
 * Rendered automatically by Next.js App Router while the chat page suspends.
 */
export default function ChatLoadingPage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-orange-50 to-white px-4">
      <LoadingState
        variant="spinner"
        message="Loading your catering assistant..."
        className="text-orange-600 [&_svg]:text-orange-600"
      />
    </div>
  );
}
