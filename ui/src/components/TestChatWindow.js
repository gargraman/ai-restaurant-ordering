'use client';

import { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import MessageList from '../MessageList';
import InputArea from '../InputArea';
import { mockChatSearch, mockGetSession, mockDeleteSession } from '@/lib/mock-api-client';
import { useChatContext } from '@/contexts/ChatContext';
import analyticsTracker from '@/lib/analytics/tracker';

const TestChatWindow = ({ sessionId }) => {
  const {
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    error,
    setError,
    updateContext,
    resetSession
  } = useChatContext();
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch session info on mount to restore context
  useEffect(() => {
    const fetchSessionInfo = async () => {
      try {
        const sessionData = await mockGetSession(sessionId);

        // Update context with session entities
        if (sessionData.entities) {
          const updates = {};

          if (sessionData.entities.location) {
            updates.location = sessionData.entities.location;
          }
          if (sessionData.entities.party_size) {
            updates.partySize = sessionData.entities.party_size;
          }
          if (sessionData.entities.dietary_restrictions) {
            updates.dietaryPreferences = sessionData.entities.dietary_restrictions;
          }

          if (Object.keys(updates).length > 0) {
            updateContext(updates);
          }
        }

      } catch (error) {
        // Fail silently - session may not exist yet
        console.warn('Could not fetch session info:', error);
      }
    };

    if (sessionId) {
      fetchSessionInfo();
    }
  }, [sessionId, updateContext]);

  const handleSend = async (userInput) => {
    if (!userInput.trim()) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: userInput,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Track the query submission
      analyticsTracker.trackQuerySubmitted(userInput, sessionId);

      // Call the mock backend API
      const response = await mockChatSearch(sessionId, userInput);

      // Add assistant message with results
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.answer,
        results: response.results,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Track zero results if applicable
      if (response.results && response.results.length === 0) {
        analyticsTracker.trackZeroResultQuery(userInput, sessionId);
      }
    } catch (err) {
      setError(err.message || 'An error occurred while processing your request.');
      analyticsTracker.trackError(err.message || 'Search API error', 'search_error', sessionId);
    } finally {
      setIsLoading(false);
    }
  };

  // Initialize with a welcome message
  useEffect(() => {
    if (messages.length === 0) {
      const welcomeMessage = {
        id: 'welcome',
        role: 'assistant',
        content: 'Hello! I\'m here to help you discover restaurants and catering options. What are you looking for today? (This is a test version using mock data)',
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
    }
  }, [messages.length, setMessages]);

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
      setIsLoading(true);

      // Delete session on backend
      await mockDeleteSession(sessionId);

      // Track analytics
      analyticsTracker.trackSessionReset(sessionId, messages.length);

      // Reset local state
      resetSession();
      setError(null);

      // Generate new session ID
      const newSessionId = uuidv4();
      localStorage.setItem('testChatSessionId', newSessionId);

      // Reload page with new session
      window.location.reload();

    } catch (error) {
      setError('Unable to reset session. Please refresh the page.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="flex flex-col h-full"
      role="main"
      aria-label="Restaurant discovery chat interface"
    >
      <MessageList messages={messages} isLoading={isLoading} error={error} sessionId={sessionId} />
      <InputArea onSend={handleSend} disabled={isLoading} />
      <div ref={messagesEndRef} />
    </div>
  );
};

export default TestChatWindow;