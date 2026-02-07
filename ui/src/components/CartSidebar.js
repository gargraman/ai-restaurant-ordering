'use client';

import { useState, useEffect } from 'react';
import { X, Plus, Minus, Trash2, CreditCard } from 'lucide-react';
import { useCart } from '@/contexts/CartContext';

const CartSidebar = ({ isOpen, onClose }) => {
  const { items, total, addItem, removeItem, updateQuantity, clearCart } = useCart();
  const [isCheckoutLoading, setIsCheckoutLoading] = useState(false);

  const handleCheckout = async () => {
    setIsCheckoutLoading(true);
    try {
      // In a real implementation, this would call the backend checkout API
      console.log('Proceeding to checkout with items:', items);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Redirect to checkout page
      window.location.href = '/checkout';
    } catch (error) {
      console.error('Checkout error:', error);
    } finally {
      setIsCheckoutLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-in panel */}
      <div className="absolute inset-y-0 right-0 max-w-full flex">
        <div className="relative w-screen max-w-md">
          <div className="h-full flex flex-col bg-white shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-4 border-b border-gray-200 sm:px-6">
              <h2 className="text-lg font-bold text-gray-900">Your Cart</h2>
              <button
                type="button"
                className="text-gray-400 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-restaurant-primary rounded-full p-1 transition-colors"
                onClick={onClose}
                aria-label="Close cart"
              >
                <X className="h-6 w-6" aria-hidden="true" />
              </button>
            </div>

            {/* Cart items */}
            <div className="flex-1 overflow-y-auto p-4">
              {items.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center py-12">
                  <div className="bg-restaurant-primary/10 rounded-full p-4 mb-4">
                    <CreditCard className="h-12 w-12 text-restaurant-primary" />
                  </div>
                  <h3 className="mt-2 text-lg font-medium text-gray-900">Your cart is empty</h3>
                  <p className="mt-1 text-sm text-gray-500">Add some delicious items to your cart!</p>
                </div>
              ) : (
                <div className="flow-root">
                  <ul className="-my-6 divide-y divide-gray-200">
                    {items.map((item) => (
                      <li key={`${item.restaurant_id}-${item.menu_item_id}`} className="py-4 flex">
                        <div className="h-16 w-16 flex-shrink-0 overflow-hidden rounded-lg border border-gray-200 sm:h-20 sm:w-20">
                          {item.image_url ? (
                            <img
                              src={item.image_url}
                              alt={item.name}
                              className="h-full w-full object-cover object-center"
                            />
                          ) : (
                            <div className="h-full w-full bg-gray-100 flex items-center justify-center">
                              <span className="text-gray-500 text-xs">No Image</span>
                            </div>
                          )}
                        </div>

                        <div className="ml-3 flex flex-1 flex-col sm:ml-4">
                          <div>
                            <div className="flex justify-between text-sm sm:text-base font-medium text-gray-900">
                              <h3 className="truncate max-w-[100px] sm:max-w-[160px]">{item.name}</h3>
                              <p className="ml-2 font-bold sm:ml-4">${(item.price * item.quantity).toFixed(2)}</p>
                            </div>
                            <p className="mt-1 text-xs sm:text-sm text-gray-500 truncate max-w-[100px] sm:max-w-[160px]">{item.restaurant_name}</p>
                          </div>
                          <div className="flex flex-1 items-end justify-between text-xs sm:text-sm mt-1 sm:mt-2">
                            <div className="flex items-center border border-gray-300 rounded-md">
                              <button
                                type="button"
                                onClick={() => updateQuantity(item.menu_item_id, Math.max(1, item.quantity - 1))}
                                className="px-2 py-1 text-gray-600 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-restaurant-primary rounded-l-md sm:px-3 sm:py-1.5"
                                aria-label="Decrease quantity"
                              >
                                <Minus className="h-3 w-3 sm:h-4 sm:w-4" />
                              </button>
                              <span className="px-2 py-1 text-gray-900 min-w-[24px] text-center sm:px-3 sm:py-1.5 sm:min-w-[32px]">{item.quantity}</span>
                              <button
                                type="button"
                                onClick={() => updateQuantity(item.menu_item_id, item.quantity + 1)}
                                className="px-2 py-1 text-gray-600 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-restaurant-primary rounded-r-md sm:px-3 sm:py-1.5"
                                aria-label="Increase quantity"
                              >
                                <Plus className="h-3 w-3 sm:h-4 sm:w-4" />
                              </button>
                            </div>

                            <button
                              type="button"
                              onClick={() => removeItem(item.menu_item_id)}
                              className="font-medium text-red-600 hover:text-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 rounded p-1"
                              aria-label="Remove item"
                            >
                              <Trash2 className="h-3 w-3 sm:h-4 sm:w-4" />
                            </button>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Footer */}
            {items.length > 0 && (
              <div className="border-t border-gray-200 p-4">
                <div className="flex justify-between text-base font-bold text-gray-900 mb-2">
                  <p>Subtotal</p>
                  <p className="text-lg">${total.toFixed(2)}</p>
                </div>
                <p className="text-xs sm:text-sm text-gray-500 mb-4">
                  Shipping and taxes calculated at checkout.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={clearCart}
                    className="bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-2.5 px-4 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 text-sm sm:text-base"
                  >
                    Clear Cart
                  </button>
                  <button
                    type="button"
                    onClick={handleCheckout}
                    disabled={isCheckoutLoading}
                    className="bg-restaurant-primary hover:bg-restaurant-primary/90 text-white font-medium py-2.5 px-4 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-restaurant-primary disabled:opacity-70 text-sm sm:text-base shadow-md hover:shadow-lg"
                  >
                    {isCheckoutLoading ? (
                      <span className="flex items-center justify-center">
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                      </span>
                    ) : 'Checkout'}
                  </button>
                </div>
                <div className="mt-3 flex justify-center">
                  <button
                    type="button"
                    onClick={onClose}
                    className="text-xs sm:text-sm font-medium text-restaurant-primary hover:text-restaurant-primary/80"
                  >
                    Continue Shopping
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CartSidebar;