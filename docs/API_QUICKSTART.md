# API Quick Start Guide

## Accessing the API Documentation

### 1. Interactive Swagger UI (Built-in)
FastAPI automatically generates interactive API documentation:

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Open in browser:
# http://localhost:8000/docs
```

**Features:**
- Try out endpoints directly from the browser
- See request/response examples
- Authentication built-in (click "Authorize" button)
- Auto-generated from code

### 2. ReDoc (Built-in Alternative)
A cleaner, read-only documentation view:

```bash
# Open in browser:
# http://localhost:8000/redoc
```

**Features:**
- Better for reading and understanding the API
- Clean layout with search
- Printable documentation

### 3. OpenAPI Spec (Custom YAML)
The comprehensive OpenAPI 3.1 spec we generated:

```bash
# View/edit the spec:
cat docs/openapi.yaml

# Validate the spec:
npx @stoplight/spectral-cli lint docs/openapi.yaml
```

## Quick Start: Common Workflows

### Workflow 1: Search for Catering Options

**No authentication required for basic search**

```bash
# 1. Start a session and search
curl -X POST http://localhost:8000/chat/search \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_demo_001",
    "user_input": "Find Italian catering in Boston for 20 people",
    "max_results": 10
  }'

# Response includes:
# - results: Array of menu items
# - answer: Natural language response
# - filters: Extracted filters (city, cuisine, etc.)
# - intent: Detected intent (search, filter, etc.)

# 2. Follow up with refinement
curl -X POST http://localhost:8000/chat/search \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_demo_001",
    "user_input": "Show me vegetarian options under $15 per person",
    "max_results": 10
  }'

# 3. Check session state
curl http://localhost:8000/session/session_demo_001
```

### Workflow 2: User Registration & Authentication

```bash
# 1. Register a new user (customer)
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "customer@example.com",
    "password": "SecurePassword123",
    "first_name": "Jane",
    "last_name": "Doe",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "role": "customer"
  }'

# Response:
# {
#   "access_token": "eyJhbGc...",
#   "refresh_token": "eyJhbGc...",
#   "token_type": "bearer",
#   "expires_in": 86400
# }

# Save the access_token for subsequent requests

# 2. Login (if already registered)
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "customer@example.com",
    "password": "SecurePassword123"
  }'

# 3. Get current user profile
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <access_token>"

# 4. Refresh token when it expires
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "<refresh_token>"
  }'
```

### Workflow 3: Add to Cart & Create Order

```bash
# Set variables
ACCESS_TOKEN="<your_access_token>"
SESSION_ID="session_demo_001"
TENANT_ID="550e8400-e29b-41d4-a716-446655440000"

# 1. Add item to cart
curl -X POST "http://localhost:8000/orders/cart/items?session_id=$SESSION_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "menu_item_id": "item_123",
    "name": "Italian Wedding Buffet",
    "unit_price_cents": 1599,
    "restaurant_id": "550e8400-0000-0000-0000-000000000001",
    "restaurant_name": "Tonys Catering",
    "quantity": 2,
    "modifiers": [],
    "special_instructions": "Extra napkins please"
  }'

# 2. View cart
curl "http://localhost:8000/orders/cart?session_id=$SESSION_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 3. Update item quantity
curl -X PATCH "http://localhost:8000/orders/cart/items/<item_id>?session_id=$SESSION_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "quantity": 3
  }'

# 4. Create order from cart (requires ENABLE_PAYMENTS=true)
curl -X POST "http://localhost:8000/orders?session_id=$SESSION_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Jane Doe",
    "customer_email": "customer@example.com",
    "customer_phone": "+1-555-123-4567",
    "fulfillment_type": "pickup",
    "tip_cents": 300,
    "notes": "Please call when ready"
  }'

# Response includes:
# - order: Order details with order_number
# - payment: Payment intent with client_secret
# Use client_secret with Stripe.js to complete payment

# 5. Track order status
curl "http://localhost:8000/orders/<order_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 6. Guest order lookup (no auth required)
curl "http://localhost:8000/orders/lookup/ORD-20240115-001?customer_email=customer@example.com"
```

### Workflow 4: Restaurant Onboarding (Restaurant Admin)

```bash
ACCESS_TOKEN="<restaurant_admin_token>"
TENANT_ID="<tenant_id>"

# 1. Create a restaurant
curl -X POST http://localhost:8000/restaurants \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Tonys Catering",
    "slug": "tonys-catering",
    "description": "Family-owned Italian catering since 1985",
    "address_line1": "123 Main St",
    "city": "Boston",
    "state": "MA",
    "postal_code": "02101",
    "phone": "+1-555-123-4567",
    "email": "orders@tonyscatering.com",
    "timezone": "America/New_York"
  }'

# Response includes restaurant_id
RESTAURANT_ID="<restaurant_id>"

# 2. Set up Stripe Connect for payments (if ENABLE_PAYMENTS=true)
curl -X POST "http://localhost:8000/restaurants/$RESTAURANT_ID/stripe/connect" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "owner@tonyscatering.com",
    "refresh_url": "https://app.example.com/restaurants/stripe/refresh",
    "return_url": "https://app.example.com/restaurants/stripe/success"
  }'

# Response includes:
# - account_id: Stripe Connect account ID
# - onboarding_url: URL to complete Stripe onboarding
# - expires_at: URL expiration time

# Redirect user to onboarding_url to complete Stripe setup

# 3. Check Stripe connection status
curl "http://localhost:8000/restaurants/$RESTAURANT_ID/stripe/status" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 4. Connect POS system (if ENABLE_POS_INTEGRATION=true)
curl -X POST "http://localhost:8000/restaurants/$RESTAURANT_ID/pos/connect" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "square",
    "credentials": {
      "access_token": "sq0atp-...",
      "merchant_id": "MERCHANT123"
    },
    "location_id": "LOCATION456"
  }'

# 5. Check POS connection status
curl "http://localhost:8000/restaurants/$RESTAURANT_ID/pos/status" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 6. Update restaurant details
curl -X PATCH "http://localhost:8000/restaurants/$RESTAURANT_ID" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "is_accepting_orders": true,
    "description": "Award-winning Italian catering"
  }'

# 7. List my restaurants
curl "http://localhost:8000/restaurants" \
  -H "Authorization: Bearer $ACCESS_TOKEN"
```

### Workflow 5: Platform Admin - Tenant Management

```bash
# Must be logged in as platform_admin
ADMIN_TOKEN="<platform_admin_token>"

# 1. Create a new tenant
curl -X POST http://localhost:8000/tenants \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Acme Catering Co",
    "slug": "acme-catering",
    "contact_email": "contact@acme-catering.com",
    "contact_phone": "+1-555-987-6543",
    "billing_email": "billing@acme-catering.com",
    "application_fee_percent": 2.5
  }'

# 2. List all tenants
curl "http://localhost:8000/tenants?limit=50&offset=0" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 3. Get specific tenant
curl "http://localhost:8000/tenants/<tenant_id>" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 4. Update tenant
curl -X PATCH "http://localhost:8000/tenants/<tenant_id>" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "application_fee_percent": 3.0,
    "is_active": true
  }'

# 5. Filter active tenants only
curl "http://localhost:8000/tenants?is_active=true" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

## Environment Setup

### Required Environment Variables

```bash
# Minimum required for basic operation
export OPENAI_API_KEY="sk-..."
export JWT_SECRET_KEY="your-secret-key-min-32-chars"

# Database connections
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5433"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export POSTGRES_DB="hybrid_search"

export REDIS_HOST="localhost"
export REDIS_PORT="6379"

export OPENSEARCH_HOST="localhost"
export OPENSEARCH_PORT="9200"
```

### Optional Features

```bash
# Enable payments (requires Stripe account)
export ENABLE_PAYMENTS=true
export STRIPE_SECRET_KEY="sk_test_..."
export STRIPE_PUBLISHABLE_KEY="pk_test_..."
export STRIPE_WEBHOOK_SECRET="whsec_..."

# Enable POS integration (requires Square account)
export ENABLE_POS_INTEGRATION=true
export SQUARE_APPLICATION_ID="sq0idp-..."
export SQUARE_APPLICATION_SECRET="sq0csp-..."
export SQUARE_ENVIRONMENT="sandbox"
export SQUARE_WEBHOOK_SIGNATURE_KEY="..."

# Enable graph search (requires Neo4j)
export ENABLE_GRAPH_SEARCH=true
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

## Testing Endpoints

### Using HTTPie (Recommended)

```bash
# Install httpie
pip install httpie

# Clean syntax for API testing
http POST localhost:8000/chat/search \
  session_id="session_123" \
  user_input="Find Italian catering in Boston"

# With authentication
http GET localhost:8000/auth/me \
  "Authorization: Bearer $ACCESS_TOKEN"
```

### Using Python Requests

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/chat/search",
    json={
        "session_id": "session_123",
        "user_input": "Find Italian catering in Boston",
        "max_results": 10
    }
)
print(response.json())

# With authentication
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get(
    "http://localhost:8000/auth/me",
    headers=headers
)
print(response.json())
```

### Using JavaScript/Fetch

```javascript
// Search
const response = await fetch('http://localhost:8000/chat/search', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    session_id: 'session_123',
    user_input: 'Find Italian catering in Boston',
    max_results: 10
  })
});
const data = await response.json();

// With authentication
const authResponse = await fetch('http://localhost:8000/auth/me', {
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});
const userData = await authResponse.json();
```

## Common Issues & Solutions

### 1. 401 Unauthorized
**Problem:** Missing or invalid JWT token

**Solution:**
```bash
# Get a fresh token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use the access_token in subsequent requests
curl http://localhost:8000/protected-endpoint \
  -H "Authorization: Bearer <access_token>"
```

### 2. 403 Forbidden
**Problem:** Insufficient permissions for the operation

**Solution:** Ensure your user has the correct role:
- Platform admin endpoints require `platform_admin` role
- Restaurant management requires `restaurant_admin` role
- Check the OpenAPI spec for required roles

### 3. 400 Validation Error
**Problem:** Invalid request data

**Solution:** Check the error details for field-specific validation errors:
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "value is not a valid email address",
      "type": "value_error.email"
    }
  ]
}
```

### 4. 500 Internal Server Error
**Problem:** Server-side error (check logs)

**Solution:**
```bash
# Check application logs
tail -f logs/app.log

# Common causes:
# - Missing OPENAI_API_KEY
# - Database connection failed
# - Redis unavailable
```

### 5. CORS Error (Browser)
**Problem:** Cross-origin request blocked

**Solution:**
```bash
# Add your frontend URL to CORS_ALLOWED_ORIGINS
export CORS_ALLOWED_ORIGINS="http://localhost:3000,https://app.example.com"

# Restart the API server
```

## Rate Limiting & Best Practices

### Search Endpoints
- Use the same `session_id` for conversational follow-ups
- Sessions expire after 24 hours of inactivity
- Maximum 500 characters per query

### Cart Management
- Cart is tied to `session_id`
- Items from different restaurants cannot be mixed
- Cart persists in Redis with session TTL

### Authentication
- Access tokens expire in 24 hours (default)
- Refresh tokens expire in 7 days (default)
- Use refresh tokens to get new access tokens
- Store tokens securely (never in localStorage for production)

### Pagination
- Default limit: 50 items
- Maximum limit: 100 items
- Use offset for pagination: `?limit=50&offset=100`

## Next Steps

1. **Explore Interactive Docs:** http://localhost:8000/docs
2. **Read Full OpenAPI Spec:** `docs/openapi.yaml`
3. **Generate SDK:** Use OpenAPI Generator for your language
4. **Check Examples:** See `docs/API_DOCUMENTATION_SUMMARY.md`

## Support Resources

- **OpenAPI Spec:** `/docs/openapi.yaml`
- **Summary Document:** `/docs/API_DOCUMENTATION_SUMMARY.md`
- **Project README:** `/CLAUDE.md`
- **Architecture Docs:** `/docs/`
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **OpenAPI 3.1 Spec:** https://spec.openapis.org/oas/v3.1.0
