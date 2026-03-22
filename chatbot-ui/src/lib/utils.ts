/**
 * Utility Functions
 */

/**
 * Generate a unique ID
 */
export function generateId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Format a Date to a readable time string (HH:MM AM/PM)
 */
export function formatTime(date: Date): string {
  const d = new Date(date);
  return d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

/**
 * Truncate text to a maximum length
 */
export function truncate(text: string, length: number): string {
  if (text.length <= length) return text;
  return text.substring(0, length) + "...";
}

/**
 * Format price as currency
 */
export function formatPrice(price: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(price);
}

/**
 * Get a random food image URL from Unsplash for menu items
 */
export function getFoodImage(): string {
  const foodImages = [
    "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1504674900769-f32dc2b92cdb?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1598103442097-8b74394b95c6?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1555939594-58d7cb561404?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=400&h=300&fit=crop",
    "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&h=300&fit=crop",
  ];
  return foodImages[Math.floor(Math.random() * foodImages.length)];
}

/**
 * Get color styling for dietary labels
 */
export function getDietaryLabelColor(label: string): string {
  const colorMap: Record<string, string> = {
    vegetarian: "bg-green-100 text-green-800 border-green-200",
    vegan: "bg-emerald-100 text-emerald-800 border-emerald-200",
    "gluten-free": "bg-amber-100 text-amber-800 border-amber-200",
    halal: "bg-blue-100 text-blue-800 border-blue-200",
    kosher: "bg-purple-100 text-purple-800 border-purple-200",
    dairy: "bg-rose-100 text-rose-800 border-rose-200",
    nut: "bg-orange-100 text-orange-800 border-orange-200",
    "soy-free": "bg-indigo-100 text-indigo-800 border-indigo-200",
    organic: "bg-lime-100 text-lime-800 border-lime-200",
  };
  return colorMap[label.toLowerCase()] || "bg-gray-100 text-gray-800 border-gray-200";
}
