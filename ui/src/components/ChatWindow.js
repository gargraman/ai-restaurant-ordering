'use client';

import { useEffect, useRef } from 'react';
import MessageList from './MessageList';
import InputArea from './InputArea';
import { chatSearch, getSession } from '@/lib/api-client';
import { useChatContext } from '@/contexts/ChatContext';
import analyticsTracker from '@/lib/analytics/tracker';

const ChatWindow = ({ sessionId }) => {
  const {
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    error,
    setError,
    updateContext
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
        const sessionData = await getSession(sessionId);

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

      // Call the backend API
      const response = await chatSearch(sessionId, userInput);

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
        content: 'Hello! I\'m here to help you discover restaurants and catering options. What are you looking for today?',
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
    }
  }, [messages.length, setMessages]);

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

export default ChatWindow;