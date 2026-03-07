import React from "react";
import Link from "next/link";
import { UtensilsCrossed } from "lucide-react";
import { Button } from "@/components/ui/button";

/**
 * Global 404 page for the Next.js App Router.
 * Shown whenever a route segment cannot be found.
 */
export default function NotFoundPage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-orange-50 to-white px-4">
      <div className="flex max-w-md flex-col items-center gap-6 text-center">
        {/* Brand icon */}
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-orange-100">
          <UtensilsCrossed
            className="h-10 w-10 text-orange-500"
            strokeWidth={1.5}
            aria-hidden="true"
          />
        </div>

        {/* Large 404 number */}
        <p
          className="bg-gradient-to-r from-orange-500 to-orange-700 bg-clip-text text-8xl font-extrabold tracking-tight text-transparent"
          aria-hidden="true"
        >
          404
        </p>

        {/* Title and description */}
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-gray-900">Page not found</h1>
          <p className="text-sm leading-relaxed text-gray-500">
            The page you&apos;re looking for doesn&apos;t exist.
          </p>
        </div>

        {/* CTA button */}
        <Button
          asChild
          className="bg-orange-600 hover:bg-orange-700 focus-visible:ring-orange-500"
        >
          <Link href="/">Back to home</Link>
        </Button>
      </div>
    </div>
  );
}
