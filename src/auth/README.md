# Authentication Module Documentation

## Overview
The authentication module provides JWT-based authentication with multi-tenant support and role-based access control (RBAC). It implements secure authentication patterns for the order management system.

## Key Features
- JWT token creation and validation (access + refresh tokens)
- Argon2 password hashing
- Role-based access control (customer, restaurant_admin, platform_admin)
- Multi-tenant support with tenant isolation
- OAuth integration for external services

## Components

### JWT Handling (`jwt.py`)
- Secure JWT token creation, validation, and refresh functionality
- Uses python-jose with security best practices
- Proper algorithm specification and expiration handling
- Secure token validation with configurable parameters

### Password Security (`password.py`)
- Argon2 password hashing (winner of Password Hashing Competition)
- Configurable parameters for security tuning
- Secure password verification
- Recommended by OWASP for secure password storage

### Authentication Dependencies (`dependencies.py`)
- Reusable FastAPI dependencies for:
  - Extracting and validating JWT tokens from requests
  - Role-based access control
  - Tenant isolation enforcement
  - Hierarchical permission checking

### OAuth Integration (`oauth.py`)
- Flexible OAuth framework supporting multiple providers
- Handles OAuth flows for Stripe, Square, Google, etc.
- OAuthConfig model for provider configuration
- Functions to generate OAuth authorization URLs
- Functions to exchange authorization codes for access tokens
- Support for multiple providers with registry pattern

### Authentication Models (`models.py`)
- Pydantic models for request/response handling
- Email validation with RFC 5322 compliant regex
- Token request/response models
- User registration and login models

## Roles and Permissions
- **PLATFORM_ADMIN**: Can access all tenants and resources
- **TENANT_ADMIN**: Can manage tenant and all restaurants
- **RESTAURANT_ADMIN**: Can manage single restaurant
- **RESTAURANT_STAFF**: Can view orders and update status
- **CUSTOMER**: End customer placing orders

## Security Measures
- Secure JWT implementation with proper algorithm specification
- Argon2 password hashing for secure storage
- Tenant isolation with RLS enforcement
- Input validation and sanitization
- Secure OAuth flows with PKCE where applicable
- Protection against common attacks (timing attacks, etc.)

## Integration Points
- Used by API routers for authentication
- Integrated with database layer for tenant isolation
- Works with session management for state tracking
- Connected to payment systems for secure transactions