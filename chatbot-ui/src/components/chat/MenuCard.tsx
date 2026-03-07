/**
 * MenuCard Component
 * Displays a menu item with image, details, and add to cart button
 */

"use client";

import React from "react";
import { motion } from "framer-motion";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { MenuItemResult } from "@/types/api";
import { formatPrice, getFoodImage, getDietaryLabelColor } from "@/lib/utils";
import { ShoppingCart } from "lucide-react";

interface MenuCardProps {
  item: MenuItemResult;
  onAddToCart: () => void;
  index?: number;
}

export function MenuCard({ item, onAddToCart, index = 0 }: MenuCardProps) {
  const image = getFoodImage();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className="h-full"
    >
      <Card className="overflow-hidden rounded-2xl border-0 shadow-xl ring-1 ring-black/5 transition-shadow hover:shadow-2xl">
        {/* Image */}
        <div className="relative h-[180px] w-full overflow-hidden">
          <img
            src={image}
            alt={item.item_name}
            className="h-full w-full object-cover transition-transform duration-300 hover:scale-110"
            loading="lazy"
          />
          {/* Gradient overlay for better contrast */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/30 via-transparent to-transparent" />
          {/* Restaurant badge — glass morphism top-right */}
          <div className="absolute right-2 top-2">
            <Badge className="border-0 bg-white/80 text-gray-900 shadow-sm backdrop-blur-sm">
              {item.restaurant_name}
            </Badge>
          </div>
        </div>

        <CardContent className="p-4">
          {/* Item Name */}
          <h3 className="mb-2 text-lg font-semibold text-gray-900">
            {item.item_name}
          </h3>

          {/* Description */}
          {item.item_description && (
            <p className="mb-3 text-sm text-gray-600 line-clamp-2">
              {item.item_description}
            </p>
          )}

          {/* Dietary Labels */}
          {item.dietary_labels && item.dietary_labels.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-1">
              {item.dietary_labels.map((label) => (
                <Badge
                  key={label}
                  variant="secondary"
                  className={`text-xs ${getDietaryLabelColor(label)}`}
                >
                  {label}
                </Badge>
              ))}
            </div>
          )}

          {/* Tags */}
          {item.tags && item.tags.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-1">
              {item.tags.slice(0, 3).map((tag) => (
                <Badge key={tag} variant="outline" className="text-xs">
                  #{tag}
                </Badge>
              ))}
            </div>
          )}

          {/* Price and Serving Info */}
          <div className="mb-3 space-y-1">
            {item.display_price && (
              <p className="text-2xl font-bold text-orange-600">
                {formatPrice(item.display_price)}
              </p>
            )}
            {(item.serves_min || item.serves_max) && (
              <p className="text-xs text-gray-500">
                Serves {item.serves_min}
                {item.serves_max ? `-${item.serves_max}` : ""} people
              </p>
            )}
            {item.price_per_person && !item.display_price && (
              <p className="text-sm text-gray-600">
                {formatPrice(item.price_per_person)} per person
              </p>
            )}
          </div>

          {/* Location */}
          {(item.city || item.state) && (
            <p className="text-xs text-gray-500">
              📍 {item.city}, {item.state}
            </p>
          )}
        </CardContent>

        <CardFooter className="border-t border-gray-100 p-3">
          <motion.div className="w-full" whileHover={{ scale: 1.02 }}>
            <Button
              onClick={onAddToCart}
              className="w-full bg-gradient-to-r from-orange-500 to-orange-600 text-white shadow-sm hover:from-orange-600 hover:to-orange-700 hover:shadow-md"
              aria-label={`Add ${item.item_name} to cart`}
            >
              <ShoppingCart className="mr-2 h-4 w-4" />
              Add to Cart
            </Button>
          </motion.div>
        </CardFooter>
      </Card>
    </motion.div>
  );
}
