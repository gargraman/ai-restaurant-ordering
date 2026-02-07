'use client';

import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { ShoppingCart, Package } from 'lucide-react';
import ChatWindow from '@/components/ChatWindow';
import CartSidebar from '@/components/CartSidebar';
import { deleteSession } from '@/lib/api-client';
import { useChatContext } from '@/contexts/ChatContext';
import { useCart } from '@/contexts/CartContext';
import analyticsTracker from '@/lib/analytics/tracker';

export default function Home() {
  // Generate a unique session ID on initial load
  const [sessionId] = useState(() => {
    if (typeof window !== 'undefined') {
      let id = localStorage.getItem('chatSessionId');
      if (!id) {
        id = uuidv4();
        localStorage.setItem('chatSessionId', id);
      }
      return id;
    }
    return '';
  });

  const { messages, resetSession } = useChatContext();
  const { itemCount } = useCart();
  const [isCartOpen, setIsCartOpen] = useState(false);

  // Function to handle session reset
  const handleNewChat = async () => {
    if (messages.length <= 1) {
      // Only welcome message, no need to confirm
      return;
    }

    const confirmed = window.confirm(
      'Start a new chat? This will clear your current conversation.'
    );

    if (!confirmed) return;

    try {
      // Delete session on backend
      await deleteSession(sessionId);

      // Track analytics
      analyticsTracker.trackSessionReset(sessionId, messages.length);

      // Reset local state
      resetSession();

      // Generate new session ID
      const newSessionId = uuidv4();
      localStorage.setItem('chatSessionId', newSessionId);

      // Reload page with new session
      window.location.reload();

    } catch (error) {
      alert('Unable to reset session. Please refresh the page.');
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-restaurant-primary/5 to-restaurant-secondary/5">
      <header className="py-3 px-4 border-b border-gray-200 bg-white shadow-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Package className="h-6 w-6 text-restaurant-primary" />
            <h1 className="text-lg font-bold text-gray-800 truncate">Restaurant Discovery Chat</h1>
          </div>
          <div className="flex items-center space-x-2">
            <a
              href="/tracking"
              className="text-sm text-gray-600 hover:text-restaurant-primary px-2 py-1.5 rounded-lg hover:bg-gray-50 transition-colors hidden sm:block"
              aria-label="Track order"
            >
              Track Order
            </a>
            <button
              onClick={() => setIsCartOpen(true)}
              className="relative text-sm text-gray-600 hover:text-restaurant-primary p-2 rounded-lg hover:bg-gray-50 transition-colors"
              aria-label="View cart"
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setIsCartOpen(true);
                }
              }}
            >
              <ShoppingCart className="h-5 w-5" />
              {itemCount > 0 && (
                <span className="absolute -top-0.5 -right-0.5 bg-restaurant-primary text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                  {itemCount}
                </span>
              )}
            </button>
            <button
              onClick={handleNewChat}
              className="text-sm text-gray-600 hover:text-restaurant-primary px-2 py-1.5 rounded-lg hover:bg-gray-50 transition-colors"
              aria-label="Start new chat"
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleNewChat();
                }
              }}
            >
              + New
            </button>
            <span className="text-xs bg-restaurant-primary/10 text-restaurant-primary px-2 py-1 rounded-full hidden sm:block">Beta</span>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden" tabIndex="0">
        <div className="h-full max-w-4xl mx-auto flex flex-col px-4 pb-2">
          <ChatWindow sessionId={sessionId} />
        </div>
      </main>

      <footer className="py-2 px-4 text-center text-xs text-gray-500 border-t border-gray-200 bg-white">
        <div className="max-w-4xl mx-auto">
          <p className="hidden sm:block">Powered by Hybrid Search v2 • Restaurant Discovery Chat</p>
          <p className="sm:hidden text-xs">Restaurant Discovery Chat</p>
          <div className="mt-1 flex flex-wrap justify-center gap-2">
            <a href="#" className="hover:text-restaurant-primary transition-colors text-xs" tabIndex="0">Privacy</a>
            <a href="#" className="hover:text-restaurant-primary transition-colors text-xs" tabIndex="0">Terms</a>
            <a href="#" className="hover:text-restaurant-primary transition-colors text-xs" tabIndex="0">Contact</a>
          </div>
        </div>
      </footer>

      <CartSidebar isOpen={isCartOpen} onClose={() => setIsCartOpen(false)} />
    </div>
  );
}