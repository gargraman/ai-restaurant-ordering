'use client';

import { format } from 'date-fns';
import RestaurantCard from './RestaurantCard';

const MessageBubble = ({ role, content, results, timestamp, sessionId }) => {
  const isUser = role === 'user';

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
      role="region"
      aria-label={`${role} message`}
    >
      <div
        className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 ${isUser ? 'bg-blue-500 text-white' : 'bg-white border border-gray-200'}`}
        role="article"
        aria-atomic="true"
      >
        <div className="whitespace-pre-wrap break-words" tabIndex="0">{content}</div>

        {results && (
          <>
            {results.length > 0 ? (
              <div className="mt-3 space-y-3">
                <p className="font-medium text-gray-700" tabIndex="0">Here are some options I found:</p>
                <div className="space-y-3" role="list" aria-label="Restaurant recommendations">
                  {results.slice(0, 5).map((result) => (
                    <div key={result.doc_id} role="listitem">
                      <RestaurantCard restaurant={result} sessionId={sessionId} />
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="mt-3">
                <p className="text-gray-600 italic" tabIndex="0">No specific restaurant options found for this query.</p>
              </div>
            )
          </>
        )}

        {timestamp && (
          <div
            className={`text-xs mt-1 ${isUser ? 'text-blue-100' : 'text-gray-500'}`}
            aria-label={`Sent at ${format(timestamp, 'h:mm a')}`}
          >
            {format(timestamp, 'h:mm a')}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;