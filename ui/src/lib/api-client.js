// API client for the restaurant discovery chatbot

// When NEXT_PUBLIC_API_URL is empty, requests use same-origin (proxied via Next.js rewrites).
// Set to a full URL (e.g. http://localhost:8000) for direct cross-origin access.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? '';

/**
 * Returns Authorization header with stored JWT token, if available.
 */
const getAuthHeaders = () => {
  if (typeof window === 'undefined') return {};
  const token = localStorage.getItem('access_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
};

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

/**
 * Add item to cart
 * @param {string} sessionId - Unique session identifier
 * @param {Object} item - Item to add to cart
 * @returns {Promise<Object>} Cart response
 */
export const addToCart = async (sessionId, item) => {
  const response = await fetch(`${API_BASE_URL}/orders/cart/items?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      menu_item_id: item.menu_item_id,
      name: item.name,
      unit_price_cents: Math.round((item.price || 0) * 100), // Convert to cents
      restaurant_id: item.restaurant_id || '',
      restaurant_name: item.restaurant_name || 'Unknown Restaurant',
      quantity: item.quantity || 1,
      modifiers: item.modifiers || [],
      special_instructions: item.special_instructions || null
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Get cart contents
 * @param {string} sessionId - Unique session identifier
 * @returns {Promise<Object>} Cart contents
 */
export const getCart = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/orders/cart?session_id=${sessionId}`);

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Update cart item quantity
 * @param {string} sessionId - Unique session identifier
 * @param {string} itemId - ID of the item to update
 * @param {number} quantity - New quantity
 * @returns {Promise<Object>} Updated cart
 */
export const updateCartItem = async (sessionId, itemId, quantity) => {
  const response = await fetch(`${API_BASE_URL}/orders/cart/items/${itemId}?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      quantity: quantity
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Remove item from cart
 * @param {string} sessionId - Unique session identifier
 * @param {string} itemId - ID of the item to remove
 * @returns {Promise<Object>} Updated cart
 */
export const removeFromCart = async (sessionId, itemId) => {
  const response = await fetch(`${API_BASE_URL}/orders/cart/items/${itemId}?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'DELETE'
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Clear cart
 * @param {string} sessionId - Unique session identifier
 * @returns {Promise<Object>} Response
 */
export const clearCart = async (sessionId) => {
  const response = await fetch(`${API_BASE_URL}/orders/cart?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'DELETE'
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  // 204 No Content — no body to parse
};

/**
 * Create order from cart
 * @param {string} sessionId - Unique session identifier
 * @param {Object} orderData - Order information
 * @returns {Promise<Object>} Order response
 */
export const createOrder = async (sessionId, orderData) => {
  const response = await fetch(`${API_BASE_URL}/orders?session_id=${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      customer_name: orderData.shipping_info.full_name,
      customer_email: orderData.shipping_info.email,
      customer_phone: orderData.shipping_info.phone,
      fulfillment_type: orderData.fulfillment_type || 'pickup',
      delivery_address: orderData.shipping_info.address ? {
        street: orderData.shipping_info.address,
        city: orderData.shipping_info.city,
        state: orderData.shipping_info.state,
        zip_code: orderData.shipping_info.zip_code
      } : null,
      notes: orderData.shipping_info.notes || '',
      tip_cents: Math.round((orderData.tip_amount || 0) * 100), // Convert to cents
      guest_identifier: orderData.guest_identifier || null
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Get order details by order number and customer email
 * @param {string} orderNumber - Order number
 * @param {string} customerEmail - Customer email for verification
 * @returns {Promise<Object>} Order details
 */
export const getOrderDetails = async (orderNumber, customerEmail) => {
  const response = await fetch(`${API_BASE_URL}/orders/lookup/${orderNumber}?customer_email=${encodeURIComponent(customerEmail)}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    }
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Login user and return tokens
 * @param {string} email - User email
 * @param {string} password - User password
 * @returns {Promise<Object>} Token response {access_token, refresh_token, token_type, expires_in}
 */
export const loginUser = async (email, password) => {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password })
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Get current authenticated user profile
 * @returns {Promise<Object>} User profile
 */
export const getCurrentUser = async () => {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Logout current user
 * @returns {Promise<void>}
 */
export const logoutUser = async () => {
  const response = await fetch(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
    },
  });

  // 204 No Content on success; ignore body
  if (!response.ok && response.status !== 204) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
};

/**
 * Create a new restaurant (admin endpoint)
 * @param {Object} restaurantData - Restaurant information
 * @returns {Promise<Object>} Created restaurant
 */
export const createRestaurant = async (restaurantData) => {
  const response = await fetch(`${API_BASE_URL}/admin/restaurants`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify(restaurantData)
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Initiate Stripe Connect onboarding for a restaurant
 * @param {string} restaurantId - Restaurant ID
 * @param {Object} connectData - {email, refresh_url, return_url}
 * @returns {Promise<Object>} {account_id, onboarding_url, expires_at}
 */
export const connectStripe = async (restaurantId, connectData) => {
  const response = await fetch(`${API_BASE_URL}/admin/restaurants/${restaurantId}/stripe/connect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify({
      email: connectData.email,
      refresh_url: connectData.refresh_url,
      return_url: connectData.return_url,
    })
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Get Stripe connection status
 * @param {string} restaurantId - Restaurant ID
 * @returns {Promise<Object>} Stripe status
 */
export const getStripeStatus = async (restaurantId) => {
  const response = await fetch(`${API_BASE_URL}/admin/restaurants/${restaurantId}/stripe/status`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    }
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Connect POS system to restaurant
 * @param {string} restaurantId - Restaurant ID
 * @param {Object} posData - POS configuration
 * @returns {Promise<Object>} Connection status
 */
export const connectPos = async (restaurantId, posData) => {
  const response = await fetch(`${API_BASE_URL}/admin/restaurants/${restaurantId}/pos/connect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    },
    body: JSON.stringify(posData)
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

/**
 * Get POS connection status
 * @param {string} restaurantId - Restaurant ID
 * @returns {Promise<Object>} POS status
 */
export const getPosStatus = async (restaurantId) => {
  const response = await fetch(`${API_BASE_URL}/admin/restaurants/${restaurantId}/pos/status`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
    }
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};