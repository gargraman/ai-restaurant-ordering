const EmptyResultsMessage = ({ query, onTryAgain }) => {
  return (
    <div className="flex justify-start">
      <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 rounded-2xl px-4 py-3">
        <p className="mb-2">I couldn't find any results for "{query}".</p>
        <p className="text-sm mb-2">Try adjusting your search criteria or asking differently.</p>
        {onTryAgain && (
          <button 
            onClick={onTryAgain}
            className="text-yellow-700 underline text-sm"
          >
            Try a different search
          </button>
        )}
      </div>
    </div>
  );
};

export default EmptyResultsMessage;