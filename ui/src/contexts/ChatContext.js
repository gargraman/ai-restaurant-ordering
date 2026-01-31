'use client';

import { createContext, useContext, useReducer } from 'react';

const ChatContext = createContext();

// Initial state
const initialState = {
  messages: [],
  isLoading: false,
  error: null,
  sessionInfo: null, // NEW: Store session metadata
  context: {
    location: '',
    partySize: null,
    dietaryPreferences: []
  }
};

// Reducer function
const chatReducer = (state, action) => {
  switch (action.type) {
    case 'SET_MESSAGES':
      return { ...state, messages: action.payload };
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload] };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload, error: null };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    case 'UPDATE_CONTEXT':
      return { ...state, context: { ...state.context, ...action.payload } };
    case 'RESET_CONTEXT':
      return { ...state, context: initialState.context };
    case 'SET_SESSION_INFO':
      return { ...state, sessionInfo: action.payload };
    case 'RESET_SESSION':
      return {
        ...initialState,
        sessionInfo: state.sessionInfo // Preserve session info if needed
      };
    default:
      return state;
  }
};

// Provider component
export const ChatProvider = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  // Actions
  const setMessages = (messages) => {
    dispatch({ type: 'SET_MESSAGES', payload: messages });
  };

  const addMessage = (message) => {
    dispatch({ type: 'ADD_MESSAGE', payload: message });
  };

  const setIsLoading = (isLoading) => {
    dispatch({ type: 'SET_LOADING', payload: isLoading });
  };

  const setError = (error) => {
    dispatch({ type: 'SET_ERROR', payload: error });
  };

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const updateContext = (contextUpdates) => {
    dispatch({ type: 'UPDATE_CONTEXT', payload: contextUpdates });
  };

  const resetContext = () => {
    dispatch({ type: 'RESET_CONTEXT' });
  };

  const setSessionInfo = (info) => {
    dispatch({ type: 'SET_SESSION_INFO', payload: info });
  };

  const resetSession = () => {
    dispatch({ type: 'RESET_SESSION' });
  };

  const value = {
    ...state,
    setMessages,
    addMessage,
    setIsLoading,
    setError,
    clearError,
    updateContext,
    resetContext,
    setSessionInfo,
    resetSession
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};

// Custom hook to use the chat context
export const useChatContext = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChatContext must be used within a ChatProvider');
  }
  return context;
};