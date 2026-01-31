const LoadingIndicator = () => {
  return (
    <div className="flex justify-start" role="status" aria-live="polite">
      <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
        <div className="flex space-x-2" aria-label="Loading">
          <div
            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
            aria-hidden="true"
          ></div>
          <div
            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-75"
            aria-hidden="true"
          ></div>
          <div
            className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"
            aria-hidden="true"
          ></div>
        </div>
        <span className="sr-only">Assistant is typing...</span>
      </div>
    </div>
  );
};

export default LoadingIndicator;