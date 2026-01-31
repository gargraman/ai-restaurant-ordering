'use client';

import { useState, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import TestChatWindow from '@/components/TestChatWindow';
import { mockDeleteSession } from '@/lib/mock-api-client';
import { useChatContext } from '@/contexts/ChatContext';
import analyticsTracker from '@/lib/analytics/tracker';

export default function TestPage() {
  // Generate a unique session ID on initial load
  const [sessionId] = useState(() => {
    if (typeof window !== 'undefined') {
      let id = localStorage.getItem('testChatSessionId');
      if (!id) {
        id = uuidv4();
        localStorage.setItem('testChatSessionId', id);
      }
      return id;
    }
    return '';
  });

  const { messages, resetSession } = useChatContext();

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
      await mockDeleteSession(sessionId);

      // Track analytics
      analyticsTracker.trackSessionReset(sessionId, messages.length);

      // Reset local state
      resetSession();

      // Generate new session ID
      const newSessionId = uuidv4();
      localStorage.setItem('testChatSessionId', newSessionId);

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
          <h1 className="text-lg sm:text-xl font-bold text-gray-800">Restaurant Discovery Chat - TEST MODE</h1>
          <div className="flex items-center space-x-3">
            <button
              onClick={handleNewChat}
              className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="Start new chat"
            >
              + New Chat
            </button>
            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full">Using Mock API</span>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        <div className="h-full max-w-4xl mx-auto flex flex-col px-4">
          <TestChatWindow sessionId={sessionId} />
        </div>
      </main>

      <footer className="py-2 px-4 sm:py-3 sm:px-6 text-center text-xs text-gray-500 border-t border-gray-200 bg-white">
        <p>Test Page - Powered by Mock API • Restaurant Discovery Chat</p>
      </footer>
    </div>
  );
}