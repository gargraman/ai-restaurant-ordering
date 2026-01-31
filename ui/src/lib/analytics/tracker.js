// Analytics utility for tracking user interactions

// Simple analytics tracker - in a real app, you might use Google Analytics, Mixpanel, etc.
class AnalyticsTracker {
  constructor() {
    // In a real implementation, initialize your analytics provider here
    // For example: ga('create', 'GA_MEASUREMENT_ID', 'auto');
  }

  // Track when a user submits a query
  trackQuerySubmitted(query, sessionId) {
    console.log('[Analytics] Query submitted:', { query, sessionId });

    // In a real implementation:
    // ga('send', 'event', 'Search', 'query_submitted', query);

    // Log to console for development
    this.logEvent('query_submitted', { query, sessionId });
  }

  // Track when a user clicks on a result
  trackResultClicked(docId, restaurantName, sessionId) {
    console.log('[Analytics] Result clicked:', { docId, restaurantName, sessionId });

    // In a real implementation:
    // ga('send', 'event', 'Search', 'result_clicked', restaurantName);

    this.logEvent('result_clicked', { docId, restaurantName, sessionId });
  }

  // Track zero-result queries
  trackZeroResultQuery(query, sessionId) {
    console.log('[Analytics] Zero result query:', { query, sessionId });

    this.logEvent('zero_result_query', { query, sessionId });
  }

  // Track when an error occurs
  trackError(errorMessage, errorType = 'general', sessionId = null) {
    console.error('[Analytics] Error tracked:', { errorMessage, errorType, sessionId });

    this.logEvent('error_occurred', { errorMessage, errorType, sessionId });
  }

  // Track session reset
  trackSessionReset(sessionId, messageCount) {
    console.log('[Analytics] Session reset:', { sessionId, messageCount });

    this.logEvent('session_reset', {
      sessionId,
      messageCount,
      timestamp: new Date().toISOString()
    });
  }

  // Track when feedback is submitted
  trackFeedbackSubmitted(docId, restaurantName, rating, sessionId) {
    console.log('[Analytics] Feedback submitted:', {
      docId,
      restaurantName,
      rating,
      sessionId
    });

    this.logEvent('feedback_submitted', {
      docId,
      restaurantName,
      rating,
      ratingType: rating >= 3 ? 'positive' : 'negative',
      sessionId,
      timestamp: new Date().toISOString()
    });
  }

  // Track when feedback submission fails
  trackFeedbackFailed(docId, errorMessage, sessionId) {
    console.error('[Analytics] Feedback failed:', {
      docId,
      errorMessage,
      sessionId
    });

    this.logEvent('feedback_failed', {
      docId,
      errorMessage,
      sessionId,
      timestamp: new Date().toISOString()
    });
  }

  // Generic event logging
  logEvent(eventName, properties = {}) {
    // In a real app, send to your analytics backend
    const event = {
      eventName,
      properties,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    // Send to a logging endpoint (would need to be implemented on the backend)
    // fetch('/api/log-event', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(event)
    // }).catch(console.error);

    // For now, just log to console
    console.log('[Analytics Event]', event);
  }
}

// Create a singleton instance
const analyticsTracker = new AnalyticsTracker();

export default analyticsTracker;