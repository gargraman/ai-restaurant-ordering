'use client';

import { useChatContext } from '@/contexts/ChatContext';

const SessionInfo = () => {
  const { context } = useChatContext();

  // Check if there's any context to display
  const hasContext = context.location || 
                     context.partySize || 
                     context.dietaryPreferences.length > 0;

  if (!hasContext) {
    return null; // Nothing to show
  }

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-4">
      <p className="text-xs text-blue-600 font-medium mb-2">Your Search Context:</p>
      <div className="flex flex-wrap gap-2">
        {context.location && (
          <span className="text-xs bg-white text-blue-800 px-2 py-1 rounded-full border border-blue-200">
            📍 {context.location}
          </span>
        )}
        {context.partySize && (
          <span className="text-xs bg-white text-blue-800 px-2 py-1 rounded-full border border-blue-200">
            👥 {context.partySize} people
          </span>
        )}
        {context.dietaryPreferences.map((pref, idx) => (
          <span key={idx} className="text-xs bg-white text-blue-800 px-2 py-1 rounded-full border border-blue-200">
            🥗 {pref}
          </span>
        ))}
      </div>
    </div>
  );
};

export default SessionInfo;