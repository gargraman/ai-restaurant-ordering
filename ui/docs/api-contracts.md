# API Contracts Expected by Frontend

## 1. Chat Search Endpoint

### Request
**Endpoint:** `POST /chat/search`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "session_id": "string (8-64 chars)",
  "user_input": "string (max 500 chars)",
  "max_results": "number (default: 10, min: 1, max: 50)"
}
```

### Response
**Success (200 OK):**
```json
{
  "session_id": "string",
  "resolved_query": "string",
  "intent": "string",
  "is_follow_up": "boolean",
  "filters": "object",
  "results": [
    {
      "doc_id": "string",
      "restaurant_name": "string",
      "city": "string",
      "state": "string",
      "item_name": "string",
      "item_description": "string | null",
      "display_price": "number | null",
      "price_per_person": "number | null",
      "serves_min": "number | null",
      "serves_max": "number | null",
      "dietary_labels": "array[string]",
      "tags": "array[string]",
      "rrf_score": "number"
    }
  ],
  "answer": "string",
  "confidence": "number",
  "processing_time_ms": "number"
}
```

**Error (500):**
```json
{
  "detail": "string"
}
```

## 2. Session Management Endpoints

### Get Session
**Endpoint:** `GET /session/{session_id}`

**Response:**
```json
{
  "session_id": "string",
  "created_at": "string (ISO date)",
  "last_activity": "string (ISO date)",
  "entities": "object",
  "conversation_length": "number",
  "previous_results_count": "number"
}
```

### Delete Session
**Endpoint:** `DELETE /session/{session_id}`

**Response:**
```json
{
  "status": "string",
  "session_id": "string"
}
```

## 3. Feedback Endpoint

### Submit Feedback
**Endpoint:** `POST /session/{session_id}/feedback`

**Body:**
```json
{
  "doc_id": "string",
  "rating": "number (1-5)",
  "comment": "string | null"
}
```

**Response:**
```json
{
  "status": "string",
  "doc_id": "string",
  "rating": "number"
}
```

## 4. Health Check Endpoint

### Health Check
**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "string",
  "timestamp": "string (ISO date)",
  "version": "string"
}
```

## 5. Frontend API Client Implementation

The frontend will need to implement the following client functions:

### Search API Client
```javascript
// Function to perform chat search
async function chatSearch(sessionId, userInput, maxResults = 10) {
  const response = await fetch('/api/chat/search', {
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
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

// Function to get session
async function getSession(sessionId) {
  const response = await fetch(`/api/session/${sessionId}`);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

// Function to delete session
async function deleteSession(sessionId) {
  const response = await fetch(`/api/session/${sessionId}`, {
    method: 'DELETE'
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

// Function to submit feedback
async function submitFeedback(sessionId, docId, rating, comment = null) {
  const response = await fetch(`/api/session/${sessionId}/feedback`, {
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
}
```