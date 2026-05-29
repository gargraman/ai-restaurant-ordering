'use client';

import { useState } from 'react';
import { StarIcon, ThumbsUpIcon, ThumbsDownIcon, CheckCircleIcon } from 'lucide-react';
import { submitFeedback } from '@/lib/api-client';
import analyticsTracker from '@/lib/analytics/tracker';

const RestaurantCard = ({ restaurant, sessionId }) => {
  const {
    doc_id,
    restaurant_name,
    city,
    state,
    item_name,
    item_description,
    display_price,
    price_per_person,
    serves_min,
    serves_max,
    dietary_labels,
    tags
  } = restaurant;

  const [feedbackStatus, setFeedbackStatus] = useState(null); // null | 'submitting' | 'submitted' | 'error'
  const [feedbackType, setFeedbackType] = useState(null); // 'positive' | 'negative'

  const handleViewMenuClick = () => {
    // Track the click event
    analyticsTracker.trackResultClicked(doc_id, restaurant_name, sessionId);

    // In a real app, this would navigate to the menu page
    console.log(`Navigating to menu for ${restaurant_name}`);
  };

  const handleFeedback = async (rating) => {
    if (feedbackStatus === 'submitted') return; // Already submitted

    setFeedbackStatus('submitting');
    setFeedbackType(rating === 5 ? 'positive' : 'negative');

    try {
      await submitFeedback(sessionId, doc_id, rating);
      setFeedbackStatus('submitted');
      analyticsTracker.trackFeedbackSubmitted(doc_id, restaurant_name, rating, sessionId);
    } catch (error) {
      setFeedbackStatus('error');
      analyticsTracker.trackFeedbackFailed(doc_id, error.message, sessionId);
    }
  };

  const renderFeedbackUI = () => {
    if (feedbackStatus === 'submitted') {
      return (
        <div className="flex items-center justify-center gap-2 text-green-700 text-sm">
          <CheckCircleIcon className="w-4 h-4" />
          <span>Thanks for your feedback!</span>
        </div>
      );
    }

    if (feedbackStatus === 'error') {
      return (
        <div className="flex items-center justify-between text-sm">
          <span className="text-red-600">Couldn't submit feedback</span>
          <button
            onClick={() => setFeedbackStatus(null)}
            className="text-red-600 underline hover:text-red-800"
          >
            Retry
          </button>
        </div>
      );
    }

    return (
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">Was this helpful?</span>
        <div className="flex gap-2">
          <button
            onClick={() => handleFeedback(5)}
            disabled={feedbackStatus === 'submitting'}
            className="p-2 rounded-lg hover:bg-green-100 transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-green-500"
            aria-label="This was helpful"
          >
            <ThumbsUpIcon className="w-4 h-4 text-green-600" />
          </button>
          <button
            onClick={() => handleFeedback(1)}
            disabled={feedbackStatus === 'submitting'}
            className="p-2 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-red-500"
            aria-label="This was not helpful"
          >
            <ThumbsDownIcon className="w-4 h-4 text-red-600" />
          </button>
        </div>
      </div>
    );
  };

  return (
    <div
      className="bg-gray-50 rounded-xl p-4 border border-gray-200 hover:border-orange-300 transition-colors w-full"
      role="article"
      aria-label={`${restaurant_name} restaurant card`}
    >
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2">
        <div className="flex-1 min-w-0">
          <h3 className="font-bold text-gray-900 truncate" tabIndex="0">{restaurant_name}</h3>
          <p className="text-sm text-gray-600">{city}, {state}</p>
        </div>
        <div
          className="flex items-center bg-orange-100 text-orange-800 px-2 py-1 rounded-full text-xs self-start"
          aria-label="Rating: 4.5 stars"
        >
          <StarIcon className="w-3 h-3 mr-1" aria-hidden="true" />
          <span>4.5</span>
        </div>
      </div>

      <div className="mt-2">
        <h4 className="font-semibold text-gray-800">{item_name}</h4>
        {item_description && (
          <p className="text-sm text-gray-600 mt-1 line-clamp-2">{item_description}</p>
        )}
      </div>

      <div className="mt-3 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2">
        <div>
          {display_price && (
            <p className="font-bold text-lg text-gray-900">${display_price.toFixed(2)}</p>
          )}
          {price_per_person && (
            <p className="text-sm text-gray-600">${price_per_person.toFixed(2)} per person</p>
          )}
          {(serves_min || serves_max) && (
            <p className="text-xs text-gray-500">
              {serves_min && `Serves ${serves_min}`}
              {serves_min && serves_max && '-'}
              {serves_max && `up to ${serves_max}`}
            </p>
          )}
        </div>

        <button
          className="bg-orange-500 hover:bg-orange-600 text-white text-xs font-medium px-3 py-1.5 rounded-lg transition-colors self-start sm:self-auto focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-orange-500"
          aria-label={`View menu for ${restaurant_name}`}
          onClick={handleViewMenuClick}
        >
          View Menu
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-1" aria-label="Dietary labels and tags">
        {dietary_labels && dietary_labels.map((label, index) => (
          <span
            key={index}
            className="text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full"
            aria-label={`Dietary label: ${label}`}
          >
            {label}
          </span>
        ))}

        {tags && tags.map((tag, index) => (
          <span
            key={index}
            className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full"
            aria-label={`Tag: ${tag}`}
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Feedback section at bottom */}
      <div className="mt-3 pt-3 border-t border-gray-200">
        {renderFeedbackUI()}
      </div>
    </div>
  );
};

export default RestaurantCard;