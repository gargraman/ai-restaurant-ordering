// Mock API client for testing the restaurant discovery chatbot UI

// Sample restaurant data
const sampleRestaurants = [
  {
    doc_id: "1",
    restaurant_name: "QDOBA Mexican Eats",
    city: "Boston",
    state: "MA",
    item_name: "Chicken Burrito Boxed Lunch",
    item_description: "Grilled chicken burrito with rice, beans, cheese, and salsa in a convenient boxed lunch format.",
    display_price: 12.25,
    price_per_person: 12.25,
    serves_min: 1,
    serves_max: 1,
    dietary_labels: ["gluten-free"],
    tags: ["popular", "quick"],
    rrf_score: 0.95
  },
  {
    doc_id: "2",
    restaurant_name: "Pasta Corner",
    city: "Cambridge",
    state: "MA",
    item_name: "Chicken Alfredo Catering Package",
    item_description: "Creamy alfredo pasta with grilled chicken, serves 10-12 people.",
    display_price: 120.00,
    price_per_person: 12.00,
    serves_min: 10,
    serves_max: 12,
    dietary_labels: ["vegetarian-option-available"],
    tags: ["catering", "family-pack"],
    rrf_score: 0.89
  },
  {
    doc_id: "3",
    restaurant_name: "Green Leaf Vegetarian",
    city: "Somerville",
    state: "MA",
    item_name: "Mediterranean Platter",
    item_description: "Assorted Mediterranean dishes including hummus, tabbouleh, falafel, and stuffed grape leaves.",
    display_price: 85.00,
    price_per_person: 8.50,
    serves_min: 8,
    serves_max: 10,
    dietary_labels: ["vegetarian", "vegan", "gluten-free-options"],
    tags: ["healthy", "organic"],
    rrf_score: 0.87
  },
  {
    doc_id: "4",
    restaurant_name: "Sushi Express",
    city: "Boston",
    state: "MA",
    item_name: "Sushi Combo for Events",
    item_description: "Selection of 50 sushi pieces perfect for small gatherings.",
    display_price: 150.00,
    price_per_person: 15.00,
    serves_min: 8,
    serves_max: 10,
    dietary_labels: ["pescatarian"],
    tags: ["premium", "events"],
    rrf_score: 0.82
  },
  {
    doc_id: "5",
    restaurant_name: "BBQ Junction",
    city: "Brookline",
    state: "MA",
    item_name: "Smoked Brisket Platter",
    item_description: "Slow-smoked brisket with sides, serves 6-8 people.",
    display_price: 95.00,
    price_per_person: 13.57,
    serves_min: 6,
    serves_max: 8,
    dietary_labels: [],
    tags: ["bbq", "meat-lovers"],
    rrf_score: 0.78
  }
];

// Simulate API delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Mock chat search function
 * @param {string} sessionId - Unique session identifier
 * @param {string} userInput - User's query
 * @param {number} maxResults - Maximum number of results to return (default: 10)
 * @returns {Promise<Object>} Search response
 */
export const mockChatSearch = async (sessionId, userInput, maxResults = 10) => {
  // Simulate network delay
  await delay(800 + Math.random() * 400);
  
  // Process the user input to determine response
  let results = [];
  let answer = "";
  
  // Simple keyword matching for demo purposes
  if (userInput.toLowerCase().includes("vegetarian")) {
    results = sampleRestaurants.filter(r => 
      r.dietary_labels.includes("vegetarian") || 
      r.restaurant_name.toLowerCase().includes("green leaf")
    ).slice(0, maxResults);
    answer = "I found some great vegetarian options for you!";
  } else if (userInput.toLowerCase().includes("chicken") || userInput.toLowerCase().includes("pasta")) {
    results = sampleRestaurants.filter(r => 
      r.item_name.toLowerCase().includes("chicken") || 
      r.item_name.toLowerCase().includes("pasta")
    ).slice(0, maxResults);
    answer = "Here are some chicken and pasta options that might interest you.";
  } else if (userInput.toLowerCase().includes("sushi")) {
    results = sampleRestaurants.filter(r => 
      r.restaurant_name.toLowerCase().includes("sushi")
    ).slice(0, maxResults);
    answer = "I found a sushi option for your event.";
  } else if (userInput.toLowerCase().includes("bbq") || userInput.toLowerCase().includes("brisket")) {
    results = sampleRestaurants.filter(r => 
      r.tags.includes("bbq") || r.item_name.toLowerCase().includes("brisket")
    ).slice(0, maxResults);
    answer = "Here are some BBQ options for your gathering.";
  } else if (userInput.toLowerCase().includes("catering")) {
    results = sampleRestaurants.filter(r => 
      r.tags.includes("catering") || r.serves_min > 5
    ).slice(0, maxResults);
    answer = "Here are some great catering options that serve larger groups.";
  } else {
    // Return random results if no specific match
    results = [...sampleRestaurants].sort(() => 0.5 - Math.random()).slice(0, maxResults);
    answer = "I found several options that might match what you're looking for!";
  }
  
  // Simulate no results scenario
  if (userInput.toLowerCase().includes("no results") || userInput.toLowerCase().includes("nothing")) {
    results = [];
    answer = "I couldn't find any options matching your request. Could you try rephrasing?";
  }
  
  // Simulate an error scenario
  if (userInput.toLowerCase().includes("error")) {
    throw new Error("Simulated API error for testing");
  }
  
  return {
    session_id: sessionId,
    resolved_query: userInput,
    intent: "search",
    is_follow_up: false,
    filters: {},
    results: results,
    answer: answer,
    confidence: 0.85,
    processing_time_ms: 125
  };
};

/**
 * Mock get session function
 * @param {string} sessionId - Unique session identifier
 * @returns {Promise<Object>} Session response
 */
export const mockGetSession = async (sessionId) => {
  await delay(200);
  
  return {
    session_id: sessionId,
    created_at: new Date().toISOString(),
    last_activity: new Date().toISOString(),
    entities: {},
    conversation_length: 3,
    previous_results_count: 5
  };
};

/**
 * Mock delete session function
 * @param {string} sessionId - Unique session identifier
 * @returns {Promise<Object>} Delete response
 */
export const mockDeleteSession = async (sessionId) => {
  await delay(200);
  
  return {
    status: "deleted",
    session_id: sessionId
  };
};

/**
 * Mock submit feedback function
 * @param {string} sessionId - Unique session identifier
 * @param {string} docId - Document ID of the result
 * @param {number} rating - Rating (1-5)
 * @param {string} comment - Optional comment
 * @returns {Promise<Object>} Feedback response
 */
export const mockSubmitFeedback = async (sessionId, docId, rating, comment = null) => {
  await delay(300);
  
  return {
    status: "received",
    doc_id: docId,
    rating: rating
  };
};