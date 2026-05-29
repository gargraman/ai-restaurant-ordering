const ErrorMessage = ({ message, onRetry }) => {
  return (
    <div className="flex justify-start" role="alert" aria-live="assertive">
      <div className="bg-red-50 border border-red-200 text-red-700 rounded-2xl px-4 py-3">
        <p className="mb-2">{message}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="text-red-700 underline text-sm focus:outline-none focus:ring-2 focus:ring-red-500 rounded"
            aria-label="Retry the failed action"
          >
            Try again
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;