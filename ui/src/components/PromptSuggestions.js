'use client';

const PromptSuggestions = ({ onSelect, className = '' }) => {
  const prompts = [
    "Find Italian catering near me",
    "Show vegetarian options",
    "Lunch for 10 people",
    "Gluten-free bakery items"
  ];

  return (
    <div
      className={`flex flex-wrap gap-2 ${className}`}
      role="group"
      aria-label="Suggested prompts"
    >
      {prompts.map((prompt, index) => (
        <button
          key={index}
          onClick={() => onSelect(prompt)}
          className="bg-white border border-gray-200 rounded-full px-3 py-1.5 text-xs sm:text-sm hover:bg-gray-50 hover:border-gray-300 transition-colors whitespace-nowrap focus:outline-none focus:ring-2 focus:ring-blue-500"
          aria-label={`Suggestion: ${prompt}`}
        >
          {prompt}
        </button>
      ))}
    </div>
  );
};

export default PromptSuggestions;