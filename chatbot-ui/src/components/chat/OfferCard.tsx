/**
 * OfferCard Component
 * Displays promotional offers and banners
 */

"use client";

import React from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Gift, Clock, ArrowRight } from "lucide-react";

interface OfferCardProps {
  title: string;
  description: string;
  discount?: string;
  validUntil?: string;
  code?: string;
  onClaim?: () => void;
  index?: number;
}

export function OfferCard({
  title,
  description,
  discount,
  validUntil,
  code,
  onClaim,
  index = 0,
}: OfferCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      whileHover={{ scale: 1.02 }}
      className="my-4"
    >
      <Card className="overflow-hidden rounded-2xl border-0 shadow-xl">
        <CardContent className="p-0">
          {/* Rich gradient background */}
          <div className="relative bg-gradient-to-br from-orange-500 via-orange-600 to-red-500 p-5">
            {/* Decorative faded circle in corner */}
            <div className="absolute -right-8 -top-8 h-40 w-40 rounded-full bg-white/10" />
            <div className="absolute -bottom-6 -right-2 h-24 w-24 rounded-full bg-white/10" />

            <div className="relative flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              {/* Left Section */}
              <div className="flex items-start gap-4">
                <div className="flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-full bg-white/20 backdrop-blur-sm">
                  <Gift className="h-7 w-7 text-white" />
                </div>
                <div className="space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-xl font-bold text-white">{title}</h3>
                    {discount && (
                      <Badge className="rounded-full bg-white px-3 py-0.5 font-bold text-orange-600 shadow-sm">
                        {discount}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-orange-100">{description}</p>

                  {/* Validity */}
                  {validUntil && (
                    <div className="flex items-center gap-1.5 text-xs text-orange-200">
                      <Clock className="h-3.5 w-3.5" />
                      <span>Valid until {validUntil}</span>
                    </div>
                  )}

                  {/* Promo Code box — glass style */}
                  {code && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-orange-200">Use code:</span>
                      <code className="rounded-lg border border-white/30 bg-white/20 px-3 py-1 font-mono text-sm font-bold tracking-widest text-white backdrop-blur-sm">
                        {code}
                      </code>
                    </div>
                  )}
                </div>
              </div>

              {/* CTA Button — white bg, orange text */}
              {onClaim && (
                <Button
                  onClick={onClaim}
                  size="lg"
                  className="shrink-0 bg-white font-semibold text-orange-600 shadow-md hover:bg-orange-50"
                >
                  Claim Offer
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
