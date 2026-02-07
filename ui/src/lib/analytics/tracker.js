// Analytics tracker utility
const analyticsTracker = {
  // Track when a query is submitted
  trackQuerySubmitted: (query, sessionId) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'search_query', {
        search_term: query,
        session_id: sessionId,
        page_title: document.title
      });
    }
    console.log('Query submitted:', { query, sessionId });
  },

  // Track when a result is clicked
  trackResultClicked: (docId, restaurantName, sessionId) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'click_result', {
        restaurant_name: restaurantName,
        doc_id: docId,
        session_id: sessionId
      });
    }
    console.log('Result clicked:', { docId, restaurantName, sessionId });
  },

  // Track when an item is added to cart
  trackAddToCart: (docId, itemName, restaurantName, price, sessionId) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'add_to_cart', {
        item_id: docId,
        item_name: itemName,
        restaurant_name: restaurantName,
        currency: 'USD',
        value: price,
        session_id: sessionId
      });
    }
    console.log('Added to cart:', { docId, itemName, restaurantName, price, sessionId });
  },

  // Track when a session is reset
  trackSessionReset: (sessionId, messageCount) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'session_reset', {
        session_id: sessionId,
        message_count: messageCount
      });
    }
    console.log('Session reset:', { sessionId, messageCount });
  },

  // Track zero result queries
  trackZeroResultQuery: (query, sessionId) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'zero_results', {
        search_term: query,
        session_id: sessionId
      });
    }
    console.log('Zero results query:', { query, sessionId });
  },

  // Track feedback submission
  trackFeedbackSubmitted: (docId, restaurantName, rating, sessionId) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'feedback_submitted', {
        item_id: docId,
        restaurant_name: restaurantName,
        rating: rating,
        session_id: sessionId
      });
    }
    console.log('Feedback submitted:', { docId, restaurantName, rating, sessionId });
  },

  // Track feedback failures
  trackFeedbackFailed: (docId, errorMessage, sessionId) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'feedback_failed', {
        item_id: docId,
        error_message: errorMessage,
        session_id: sessionId
      });
    }
    console.error('Feedback failed:', { docId, errorMessage, sessionId });
  },

  // Track general errors
  trackError: (errorMessage, errorType, sessionId) => {
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'error_occurred', {
        error_message: errorMessage,
        error_type: errorType,
        session_id: sessionId
      });
    }
    console.error('Error tracked:', { errorMessage, errorType, sessionId });
  }
};

export default analyticsTracker;