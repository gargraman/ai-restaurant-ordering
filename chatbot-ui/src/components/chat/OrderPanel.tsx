/**
 * OrderPanel Component
 * Displays order summary with collapsible functionality
 */

"use client";

import React from "react";
import { motion } from "framer-motion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card";
import type { CartItem } from "@/types/api";
import { formatPrice } from "@/lib/utils";
import {
  ShoppingBag,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Plus,
  Minus,
  Lock,
} from "lucide-react";

interface OrderPanelProps {
  items: CartItem[];
  onUpdateQuantity: (itemId: string, quantity: number) => void;
  onRemoveItem: (itemId: string) => void;
  onCheckout?: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export function OrderPanel({
  items,
  onUpdateQuantity,
  onRemoveItem,
  onCheckout,
  isCollapsed = false,
  onToggleCollapse,
}: OrderPanelProps) {
  // Calculate totals
  const subtotal = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );
  const tax = subtotal * 0.08; // 8% tax
  const total = subtotal + tax;

  const itemCount = items.reduce((sum, item) => sum + item.quantity, 0);

  if (isCollapsed) {
    return (
      <motion.div
        initial={{ width: 0, opacity: 0 }}
        animate={{ width: "auto", opacity: 1 }}
        exit={{ width: 0, opacity: 0 }}
        className="flex-shrink-0"
      >
        {/* Slim orange accent bar with expand button */}
        <div className="relative flex h-full items-center">
          <div className="absolute left-0 top-0 h-full w-1 rounded-r bg-gradient-to-b from-orange-500 to-orange-600" />
          <Button
            onClick={() => onToggleCollapse?.()}
            variant="ghost"
            className="h-full rounded-l-none pl-5 pr-3 text-orange-600 hover:bg-orange-50"
            aria-label="Expand order panel"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ x: 400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 400, opacity: 0 }}
      transition={{ type: "spring", damping: 20 }}
      className="flex h-full w-full flex-shrink-0 flex-col border-l bg-white lg:w-[380px]"
      role="complementary"
      aria-label="Order summary"
    >
      {/* Header — white bg, subtle border, orange icon */}
      <CardHeader className="border-b border-gray-100 bg-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <ShoppingBag className="h-5 w-5 text-orange-600" />
            <CardTitle className="text-lg font-bold">Your Order</CardTitle>
            {itemCount > 0 && (
              <Badge className="rounded-full bg-gradient-to-r from-orange-500 to-orange-600 px-2.5 text-white">
                {itemCount}
              </Badge>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onToggleCollapse?.()}
            className="text-gray-400 hover:bg-orange-50 hover:text-orange-600"
            aria-label="Collapse order panel"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      {/* Order Items */}
      <CardContent className="flex-1 overflow-hidden p-0">
        {items.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 p-8 text-center">
            <ShoppingBag className="h-14 w-14 text-gray-200" />
            <div>
              <p className="text-base font-semibold text-gray-800">
                Your cart is empty
              </p>
              <p className="mt-1 text-sm text-gray-400">
                Add items from the chat to see them here
              </p>
            </div>
          </div>
        ) : (
          <ScrollArea className="h-[calc(100vh-300px)]">
            <div className="space-y-3 p-4">
              {items.map((item, index) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -100 }}
                  transition={{ delay: index * 0.05 }}
                  className="flex gap-3 rounded-xl border border-gray-100 bg-white p-3 shadow-sm"
                >
                  {/* Item Image */}
                  <div className="h-16 w-16 flex-shrink-0 overflow-hidden rounded-lg">
                    <img
                      src={item.image}
                      alt={item.name}
                      className="h-full w-full object-cover"
                    />
                  </div>

                  {/* Item Details */}
                  <div className="flex flex-1 flex-col gap-1.5">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 pr-1">
                        <h4 className="text-sm font-semibold leading-tight text-gray-900">
                          {item.name}
                        </h4>
                        <p className="text-xs text-gray-400">
                          {item.restaurantName}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 flex-shrink-0 text-red-400 hover:bg-red-50 hover:text-red-600"
                        onClick={() => onRemoveItem(item.id)}
                        aria-label={`Remove ${item.name}`}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>

                    <div className="flex items-center justify-between">
                      {/* Circular +/- stepper */}
                      <div className="flex items-center gap-1.5">
                        <Button
                          variant="outline"
                          size="icon"
                          className="h-6 w-6 rounded-full border-gray-200 text-gray-600 hover:border-orange-400 hover:text-orange-600"
                          onClick={() =>
                            onUpdateQuantity(item.id, item.quantity - 1)
                          }
                          disabled={item.quantity <= 1}
                        >
                          <Minus className="h-2.5 w-2.5" />
                        </Button>
                        <span className="w-5 text-center text-sm font-semibold text-gray-800">
                          {item.quantity}
                        </span>
                        <Button
                          variant="outline"
                          size="icon"
                          className="h-6 w-6 rounded-full border-gray-200 text-gray-600 hover:border-orange-400 hover:text-orange-600"
                          onClick={() =>
                            onUpdateQuantity(item.id, item.quantity + 1)
                          }
                        >
                          <Plus className="h-2.5 w-2.5" />
                        </Button>
                      </div>
                      <span className="text-sm font-bold text-gray-900">
                        {formatPrice(item.price * item.quantity)}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </ScrollArea>
        )}
      </CardContent>

      {/* Footer with Totals */}
      {items.length > 0 && (
        <CardFooter className="border-t border-gray-100 bg-white p-4">
          <div className="w-full space-y-4">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-gray-500">
                <span>Subtotal</span>
                <span>{formatPrice(subtotal)}</span>
              </div>
              <div className="flex justify-between text-gray-500">
                <span>Tax (8%)</span>
                <span>{formatPrice(tax)}</span>
              </div>
              <div className="flex justify-between border-t border-gray-100 pt-2.5 text-lg font-bold text-gray-900">
                <span>Total</span>
                <span>{formatPrice(total)}</span>
              </div>
            </div>

            <Button
              onClick={onCheckout}
              className="w-full bg-gradient-to-r from-orange-500 to-orange-600 text-white shadow-md hover:from-orange-600 hover:to-orange-700 hover:shadow-lg"
              size="lg"
            >
              <Lock className="mr-2 h-4 w-4" />
              Proceed to Checkout
            </Button>
          </div>
        </CardFooter>
      )}
    </motion.div>
  );
}
