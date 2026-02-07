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
    <div className="flex flex-col h-screen bg-gradient-to-br from-orange-50 to-amber-50">
      <header className="py-3 px-4 sm:py-4 sm:px-6 border-b border-gray-200 bg-white shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <h1 className="text-lg sm:text-xl font-bold text-gray-800">Restaurant Discovery Chat</h1>
          <div className="flex items-center space-x-3">
            <a
              href="/tracking"
              className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="Track order"
            >
              Track Order
            </a>
            <button
              onClick={() => setIsCartOpen(true)}
              className="relative text-sm text-gray-600 hover:text-gray-900 p-2 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="View cart"
            >
              <ShoppingCart className="h-5 w-5" />
              {itemCount > 0 && (
                <span className="absolute top-0 right-0 bg-orange-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {itemCount}
                </span>
              )}
            </button>
            <button
              onClick={handleNewChat}
              className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="Start new chat"
            >
              + New Chat
            </button>
            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">Beta</span>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        <div className="h-full max-w-4xl mx-auto flex flex-col px-4">
          <ChatWindow sessionId={sessionId} />
        </div>
      </main>

      <footer className="py-2 px-4 sm:py-3 sm:px-6 text-center text-xs text-gray-500 border-t border-gray-200 bg-white">
        <p>Powered by Hybrid Search v2 • Restaurant Discovery Chat</p>
      </footer>

      <CartSidebar isOpen={isCartOpen} onClose={() => setIsCartOpen(false)} />
    </div>
  );
}