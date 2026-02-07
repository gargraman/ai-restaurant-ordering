'use client';

import { useState } from 'react';
import { SendIcon } from 'lucide-react';
import PromptSuggestions from './PromptSuggestions';

const InputArea = ({ onSend, disabled }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !disabled) {
      onSend(inputValue);
      setInputValue('');
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInputValue(suggestion);
    onSend(suggestion);
  };

  return (
    <div className="p-4 border-t border-gray-200 bg-white">
      <div
        className="mb-3 max-h-20 overflow-y-auto"
        aria-label="Suggested prompts"
      >
        <PromptSuggestions onSelect={handleSuggestionClick} />
      </div>
      <form
        onSubmit={handleSubmit}
        className="flex flex-col sm:flex-row items-stretch gap-2"
        role="form"
        aria-label="Chat input form"
      >
        <div className="relative flex-1">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about restaurants, cuisine, or catering..."
            disabled={disabled}
            className="w-full border border-gray-300 rounded-full pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-restaurant-primary disabled:bg-gray-100 min-h-[48px] transition-all duration-200"
            aria-label="Type your message"
            aria-disabled={disabled}
            autoComplete="off"
            aria-describedby={inputValue ? "send-button" : undefined}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          {inputValue && (
            <button
              type="button"
              onClick={() => setInputValue('')}
              className="absolute right-10 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-restaurant-primary rounded-full p-1"
              aria-label="Clear input"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </button>
          )}
        </div>
        <button
          id="send-button"
          type="submit"
          disabled={!inputValue.trim() || disabled}
          className="bg-restaurant-primary hover:bg-restaurant-primary/90 text-white rounded-full p-3 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-h-[48px] min-w-[48px] focus:outline-none focus:ring-2 focus:ring-restaurant-primary shadow-md hover:shadow-lg"
          aria-label="Send message"
          aria-disabled={disabled || !inputValue.trim()}
        >
          <SendIcon className="w-5 h-5" aria-hidden="true" />
        </button>
      </form>
    </div>
  );
};

export default InputArea;