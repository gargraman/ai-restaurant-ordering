const LoadingIndicator = () => {
  return (
    <div className="flex justify-start animate-in" role="status" aria-live="polite">
      <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
        <div className="flex items-center space-x-2" aria-label="Assistant is typing">
          <div
            className="w-2 h-2 bg-restaurant-primary rounded-full animate-bounce"
            style={{ animationDelay: '0ms' }}
            aria-hidden="true"
          ></div>
          <div
            className="w-2 h-2 bg-restaurant-primary rounded-full animate-bounce"
            style={{ animationDelay: '300ms' }}
            aria-hidden="true"
          ></div>
          <div
            className="w-2 h-2 bg-restaurant-primary rounded-full animate-bounce"
            style={{ animationDelay: '600ms' }}
            aria-hidden="true"
          ></div>
        </div>
        <span className="sr-only">Assistant is typing...</span>
      </div>
    </div>
  );
};

export default LoadingIndicator;