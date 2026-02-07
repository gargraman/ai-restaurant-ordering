'use client';

import { useState } from 'react';
import { StarIcon, ThumbsUpIcon, ThumbsDownIcon, CheckCircleIcon, ShoppingCart } from 'lucide-react';
import { submitFeedback } from '@/lib/api-client';
import analyticsTracker from '@/lib/analytics/tracker';
import { useCart } from '@/contexts/CartContext';

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

  const { addItem } = useCart();

  const handleViewMenuClick = () => {
    // Track the click event
    analyticsTracker.trackResultClicked(doc_id, restaurant_name, sessionId);

    // In a real app, this would navigate to the menu page
    console.log(`Navigating to menu for ${restaurant_name}`);
  };

  const handleAddToCart = () => {
    // Add item to cart
    const cartItem = {
      menu_item_id: doc_id,
      name: item_name,
      restaurant_name: restaurant_name,
      price: display_price || price_per_person || 0,
      restaurant_id: restaurant.restaurant_id || '', // Using a fallback if not provided
      quantity: 1
    };

    addItem(cartItem);

    // Track the add to cart event
    analyticsTracker.trackAddToCart(doc_id, item_name, restaurant_name, display_price || price_per_person || 0, sessionId);
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
        <span className="text-xs text-gray-500 font-medium">Was this helpful?</span>
        <div className="flex gap-1">
          <button
            onClick={() => handleFeedback(5)}
            disabled={feedbackStatus === 'submitting'}
            className="p-2 rounded-lg hover:bg-green-100 transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-green-500 text-green-600"
            aria-label="This was helpful"
          >
            <ThumbsUpIcon className="w-4 h-4" />
          </button>
          <button
            onClick={() => handleFeedback(1)}
            disabled={feedbackStatus === 'submitting'}
            className="p-2 rounded-lg hover:bg-red-100 transition-colors disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-red-500 text-red-600"
            aria-label="This was not helpful"
          >
            <ThumbsDownIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  };

  return (
    <div
      className="bg-white rounded-xl p-4 border border-gray-200 hover:border-restaurant-primary/50 transition-colors w-full shadow-sm hover:shadow-md"
      role="article"
      aria-label={`${restaurant_name} restaurant card`}
    >
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2">
        <div className="flex-1 min-w-0">
          <h3 className="font-bold text-lg text-gray-900 truncate" tabIndex="0">{restaurant_name}</h3>
          <p className="text-sm text-gray-600 mt-1">{city}, {state}</p>
        </div>
        <div
          className="flex items-center bg-restaurant-primary/10 text-restaurant-primary px-2.5 py-1 rounded-full text-xs self-start"
          aria-label="Rating: 4.5 stars"
        >
          <StarIcon className="w-3.5 h-3.5 mr-1" aria-hidden="true" />
          <span>4.5</span>
        </div>
      </div>

      <div className="mt-3">
        <h4 className="font-semibold text-gray-800 text-base">{item_name}</h4>
        {item_description && (
          <p className="text-sm text-gray-600 mt-2 line-clamp-2">{item_description}</p>
        )}
      </div>

      <div className="mt-4 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3">
        <div>
          {display_price && (
            <p className="font-bold text-lg sm:text-xl text-gray-900">${display_price.toFixed(2)}</p>
          )}
          {price_per_person && (
            <p className="text-sm text-gray-600">${price_per_person.toFixed(2)} per person</p>
          )}
          {(serves_min || serves_max) && (
            <p className="text-xs text-gray-500 mt-1">
              {serves_min && `Serves ${serves_min}`}
              {serves_min && serves_max && ' - '}
              {serves_max && `up to ${serves_max}`}
            </p>
          )}
        </div>

        <div className="flex gap-2 flex-wrap justify-end">
          <button
            className="bg-restaurant-primary hover:bg-restaurant-primary/90 text-white text-sm font-medium px-3 py-2 rounded-lg transition-colors self-start sm:self-auto focus:outline-none focus:ring-2 focus:ring-restaurant-primary focus:ring-offset-2 min-h-[36px] w-full sm:w-auto"
            aria-label={`View menu for ${restaurant_name}`}
            onClick={handleViewMenuClick}
          >
            View Menu
          </button>

          <button
            className="bg-gray-800 hover:bg-gray-900 text-white text-sm font-medium p-2 rounded-lg transition-colors self-start sm:self-auto focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 min-h-[36px]"
            aria-label={`Add ${item_name} to cart`}
            onClick={handleAddToCart}
          >
            <ShoppingCart className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-1.5" aria-label="Dietary labels and tags">
        {dietary_labels && dietary_labels.map((label, index) => (
          <span
            key={index}
            className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full inline-flex items-center"
            aria-label={`Dietary label: ${label}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-2.5 w-2.5 mr-1" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
            {label}
          </span>
        ))}

        {tags && tags.map((tag, index) => (
          <span
            key={index}
            className="text-xs bg-restaurant-accent/10 text-restaurant-accent px-2 py-1 rounded-full inline-flex items-center"
            aria-label={`Tag: ${tag}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-2.5 w-2.5 mr-1" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M17.707 9.293a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-7-7A.997.997 0 012 10V5a3 3 0 013-3h5c.256 0 .512.098.707.293l7 7zM5 6a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
            {tag}
          </span>
        ))}
      </div>

      {/* Feedback section at bottom */}
      <div className="mt-4 pt-3 border-t border-gray-100">
        {renderFeedbackUI()}
      </div>
    </div>
  );
};

export default RestaurantCard;