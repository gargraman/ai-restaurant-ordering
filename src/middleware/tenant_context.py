"""Tenant context middleware for RLS enforcement."""
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth.jwt import get_jwt_service, TokenError


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Extract tenant context from JWT and store in request state.
    
    This middleware:
    1. Extracts JWT token from Authorization header
    2. Decodes and validates the token
    3. Stores user_id, tenant_id, restaurant_id, role in request.state
    4. These values are used for RLS enforcement in database operations
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request and extract tenant context."""
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
            try:
                # Decode JWT token using JWT service
                jwt_service = get_jwt_service()
                token_data = jwt_service.decode_token(token)

                # Store in request state for access in route handlers
                request.state.user_id = token_data.sub
                request.state.email = token_data.email
                request.state.role = token_data.role
                request.state.tenant_id = token_data.tenant_id
                request.state.restaurant_id = token_data.restaurant_id

                # Flag for authenticated request
                request.state.authenticated = True

            except TokenError:
                # Invalid token - continue without context
                # Route handlers will enforce authentication if needed
                request.state.authenticated = False
        else:
            # No token provided
            request.state.authenticated = False
        
        # Continue processing request
        response = await call_next(request)
        return response
