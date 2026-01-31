// API client for the restaurant discovery chatbot

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Perform a chat search
 * @param {string} sessionId - Unique session identifier
 * @param {string} userInput - User's query
 * @param {number} maxResults - Maximum number of results to return (default: 10)
 * @returns {Promise<Object>} Search response
 */
export const chatSearch = async (sessionId, userInput, maxResults = 10) => {
  const response = await fetch(`${API_BASE_URL}/chat/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      user_input: userInput,
      max_results: maxResults
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Get session details
 * @param {string} sessionId - Unique session identifier
 * @returns {Promise<Object>} Session response
 */
export const getSession = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Delete session
 * @param {string} sessionId - Unique session identifier
 * @returns {Promise<Object>} Delete response
 */
export const deleteSession = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}`, {
    method: 'DELETE'
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Submit feedback for a result
 * @param {string} sessionId - Unique session identifier
 * @param {string} docId - Document ID of the result
 * @param {number} rating - Rating (1-5)
 * @param {string} comment - Optional comment
 * @returns {Promise<Object>} Feedback response
 */
export const submitFeedback = async (sessionId, docId, rating, comment = null) => {
  const response = await fetch(`${API_BASE_URL}/session/${sessionId}/feedback`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      doc_id: docId,
      rating: rating,
      comment: comment
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
};