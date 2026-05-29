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
        className="flex items-center space-x-2"
        role="form"
        aria-label="Chat input form"
      >
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask about restaurants, cuisine, or catering..."
          disabled={disabled}
          className="flex-1 border border-gray-300 rounded-full px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 min-h-[44px]"
          aria-label="Type your message"
          aria-disabled={disabled}
          autoComplete="off"
        />
        <button
          type="submit"
          disabled={!inputValue.trim() || disabled}
          className="bg-blue-500 hover:bg-blue-600 text-white rounded-full p-3 disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-h-[44px] min-w-[44px] focus:outline-none focus:ring-2 focus:ring-blue-500"
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