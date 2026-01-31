'use client';

import MessageBubble from './MessageBubble';
import LoadingIndicator from './LoadingIndicator';
import ErrorMessage from './ErrorMessage';
import EmptyResultsMessage from './EmptyResultsMessage';
import SessionInfo from './SessionInfo';

const MessageList = ({ messages, isLoading, error, sessionId }) => {
  return (
    <div
      className="flex-1 overflow-y-auto p-4 space-y-4 pb-2"
      role="log"
      aria-live="polite"
      aria-label="Chat messages"
    >
      <SessionInfo />
      {messages.map((message) => (
        <MessageBubble
          key={message.id}
          role={message.role}
          content={message.content}
          results={message.results}
          timestamp={message.timestamp}
          sessionId={sessionId}
        />
      ))}

      {error && <ErrorMessage message={error} />}

      {isLoading && <LoadingIndicator />}

      {/* Scroll anchor */}
      <div className="h-4" aria-hidden="true"></div>
    </div>
  );
};

export default MessageList;