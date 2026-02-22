'use client';

import { createContext, useContext, useReducer, useEffect, useState } from 'react';
import { getCart, addToCart, updateCartItem, removeFromCart, clearCart as clearCartApi } from '@/lib/api-client';
import { mockGetCart, mockAddToCart, mockUpdateCartItem, mockRemoveFromCart, mockClearCart } from '@/lib/mock-cart-api';

const CartContext = createContext();

// Initial state
const initialState = {
  items: [],
  total: 0,
  itemCount: 0,
  loading: false,
  error: null
};

// Reducer function
const cartReducer = (state, action) => {
  switch (action.type) {
    case 'SET_LOADING':
      return {
        ...state,
        loading: action.payload
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        loading: false
      };

    case 'SET_CART':
      // Convert cents to dollars for frontend display
      const convertedItems = action.payload.items.map(item => ({
        ...item,
        price: item.unit_price_cents / 100, // Convert cents to dollars
        total: (item.unit_price_cents * item.quantity) / 100 // Convert cents to dollars
      }));

      const total = convertedItems.reduce(
        (sum, item) => sum + (item.unit_price_cents * item.quantity) / 100, 0
      );
      const itemCount = convertedItems.reduce(
        (count, item) => count + item.quantity, 0
      );

      return {
        ...state,
        items: convertedItems,
        total,
        itemCount,
        loading: false,
        error: null
      };

    case 'ADD_ITEM':
      // Since we're now using SET_CART after API calls, this action might not be needed
      // But keeping it for fallback purposes
      const existingItemIndex = state.items.findIndex(
        item => item.menu_item_id === action.payload.menu_item_id
      );

      if (existingItemIndex >= 0) {
        const updatedItems = [...state.items];
        updatedItems[existingItemIndex] = {
          ...updatedItems[existingItemIndex],
          quantity: updatedItems[existingItemIndex].quantity + (action.payload.quantity || 1)
        };

        const newTotal = updatedItems.reduce(
          (sum, item) => sum + item.total, 0
        );

        return {
          ...state,
          items: updatedItems,
          total: newTotal,
          itemCount: updatedItems.reduce((count, item) => count + item.quantity, 0),
          loading: false
        };
      } else {
        // Convert the new item to match backend response format
        const newItem = {
          id: action.payload.id || `temp_${Date.now()}`,
          menu_item_id: action.payload.menu_item_id,
          name: action.payload.name,
          quantity: action.payload.quantity || 1,
          unit_price_cents: Math.round((action.payload.price || 0) * 100),
          total_cents: Math.round((action.payload.price || 0) * 100) * (action.payload.quantity || 1),
          modifiers: action.payload.modifiers || [],
          special_instructions: action.payload.special_instructions || null,
          // For frontend display
          price: action.payload.price || 0,
          total: (action.payload.price || 0) * (action.payload.quantity || 1)
        };

        const updatedItems = [...state.items, newItem];
        const newTotal = updatedItems.reduce(
          (sum, item) => sum + item.total, 0
        );

        return {
          ...state,
          items: updatedItems,
          total: newTotal,
          itemCount: updatedItems.reduce((count, item) => count + item.quantity, 0),
          loading: false
        };
      }

    case 'REMOVE_ITEM':
      const filteredItems = state.items.filter(
        item => item.menu_item_id !== action.payload
      );

      const remainingTotal = filteredItems.reduce(
        (sum, item) => sum + item.total, 0
      );

      return {
        ...state,
        items: filteredItems,
        total: remainingTotal,
        itemCount: filteredItems.reduce((count, item) => count + item.quantity, 0),
        loading: false
      };

    case 'UPDATE_QUANTITY':
      const updatedItems = state.items.map(item => {
        if (item.menu_item_id === action.payload.itemId) {
          const newQuantity = action.payload.quantity;
          return {
            ...item,
            quantity: newQuantity,
            total_cents: item.unit_price_cents * newQuantity, // Update total in cents
            total: (item.unit_price_cents / 100) * newQuantity // Update total in dollars
          };
        }
        return item;
      }).filter(item => item.quantity > 0); // Remove items with quantity 0

      const newTotal = updatedItems.reduce(
        (sum, item) => sum + item.total, 0
      );

      return {
        ...state,
        items: updatedItems,
        total: newTotal,
        itemCount: updatedItems.reduce((count, item) => count + item.quantity, 0),
        loading: false
      };

    case 'CLEAR_CART':
      return {
        ...initialState
      };

    default:
      return state;
  }
};

// Provider component
export const CartProvider = ({ children }) => {
  const [state, dispatch] = useReducer(cartReducer, initialState);
  const [sessionId, setSessionId] = useState('');

  // Read sessionId from localStorage client-side only
  useEffect(() => {
    setSessionId(localStorage.getItem('chatSessionId') || '');
  }, []);

  // Load cart from API on mount
  useEffect(() => {
    const loadCart = async () => {
      if (!sessionId) return;

      dispatch({ type: 'SET_LOADING', payload: true });
      try {
        const cartData = await getCart(sessionId);
        dispatch({ type: 'SET_CART', payload: cartData });
      } catch (error) {
        console.warn('Failed to load cart from backend, using mock data:', error.message);
        // Try to load from mock API
        try {
          const mockCartData = await mockGetCart(sessionId);
          dispatch({ type: 'SET_CART', payload: mockCartData });
        } catch (mockError) {
          console.error('Failed to load mock cart data:', mockError.message);
          dispatch({ type: 'SET_ERROR', payload: mockError.message });
        }
      }
    };

    loadCart();
  }, [sessionId]);

  // Actions
  const addItem = async (item) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const cartData = await addToCart(sessionId, item);
      // Update the cart with the response data to ensure consistency
      dispatch({ type: 'SET_CART', payload: cartData });
    } catch (error) {
      console.warn('Failed to add item to cart via backend, using mock:', error.message);
      // Try to use mock API
      try {
        const mockCartData = await mockAddToCart(sessionId, item);
        dispatch({ type: 'SET_CART', payload: mockCartData.cart || mockCartData });
      } catch (mockError) {
        dispatch({ type: 'SET_ERROR', payload: mockError.message });
      }
    }
  };

  const removeItem = async (itemId) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const cartData = await removeFromCart(sessionId, itemId);
      // Update the cart with the response data to ensure consistency
      dispatch({ type: 'SET_CART', payload: cartData });
    } catch (error) {
      console.warn('Failed to remove item from cart via backend, using mock:', error.message);
      // Try to use mock API
      try {
        const mockCartData = await mockRemoveFromCart(sessionId, itemId);
        dispatch({ type: 'SET_CART', payload: mockCartData.cart || mockCartData });
      } catch (mockError) {
        dispatch({ type: 'SET_ERROR', payload: mockError.message });
      }
    }
  };

  const updateQuantity = async (itemId, quantity) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const cartData = await updateCartItem(sessionId, itemId, quantity);
      // Update the cart with the response data to ensure consistency
      dispatch({ type: 'SET_CART', payload: cartData });
    } catch (error) {
      console.warn('Failed to update cart item via backend, using mock:', error.message);
      // Try to use mock API
      try {
        const mockCartData = await mockUpdateCartItem(sessionId, itemId, quantity);
        dispatch({ type: 'SET_CART', payload: mockCartData.cart || mockCartData });
      } catch (mockError) {
        dispatch({ type: 'SET_ERROR', payload: mockError.message });
      }
    }
  };

  const clearCart = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      await clearCartApi(sessionId);
      dispatch({ type: 'CLEAR_CART' });
    } catch (error) {
      console.warn('Failed to clear cart via backend, using mock:', error.message);
      // Try to use mock API
      try {
        await mockClearCart(sessionId);
        dispatch({ type: 'CLEAR_CART' });
      } catch (mockError) {
        dispatch({ type: 'SET_ERROR', payload: mockError.message });
      }
    }
  };

  const value = {
    ...state,
    addItem,
    removeItem,
    updateQuantity,
    clearCart
  };

  return (
    <CartContext.Provider value={value}>
      {children}
    </CartContext.Provider>
  );
};

// Custom hook to use the cart context
export const useCart = () => {
  const context = useContext(CartContext);
  if (!context) {
    throw new Error('useCart must be used within a CartProvider');
  }
  return context;
};