const ErrorMessage = ({ message, onRetry }) => {
  return (
    <div className="flex justify-start animate-in" role="alert" aria-live="assertive">
      <div className="bg-destructive/10 border border-destructive/30 text-destructive rounded-2xl px-4 py-3 shadow-sm w-full max-w-3xl">
        <div className="flex items-start">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 mt-0.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div className="flex-1">
            <p className="mb-2">{message}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="inline-flex items-center text-destructive hover:text-destructive/90 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-destructive/50 rounded px-2 py-1 -mx-2"
                aria-label="Retry the failed action"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                </svg>
                Try again
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ErrorMessage;