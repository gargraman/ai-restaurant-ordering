# Configuration Module Documentation

## Overview
The configuration module provides application settings and configuration management using Pydantic Settings. It handles environment variables, validation, and centralized configuration for the entire application.

## Architecture
- **Settings Model**: Centralized configuration using Pydantic BaseSettings
- **Environment Loading**: Automatic loading from .env files
- **Validation**: Built-in validation for required configurations
- **Caching**: Cached settings instance for performance

## Key Components

### Settings Model (`settings.py`)
- Application-wide configuration using Pydantic BaseSettings
- Environment variable loading with .env file support
- Type validation and coercion
- Default values for all configuration options
- Property methods for derived configurations
- Validation methods for required configurations

### Logging Configuration (`logging.py`)
- Structured logging setup using structlog
- Console and file logging configuration
- Log level management
- Structured log format for better analysis
- Performance-optimized logging configuration

## Configuration Categories

### Application Settings
- Application name and environment
- Debug and logging levels
- Host and port configurations
- Feature flags for experimental features

### OpenAI Settings
- API key for OpenAI services
- Model selection for chat and embeddings
- Embedding dimensions configuration
- Rate limiting and retry settings

### OpenSearch Settings
- Host and port configuration
- Authentication credentials
- Index name configuration
- SSL/TLS settings
- Connection pooling parameters

### PostgreSQL/pgvector Settings
- Database connection parameters
- Async connection string generation
- Connection pooling configuration
- Schema and table settings

### Redis Settings
- Redis connection parameters
- Authentication and security
- Session TTL configuration
- Connection pooling settings

### Stripe Payment Settings
- Secret and publishable keys
- Webhook secret for verification
- API version configuration
- Processing fee settings

### Square POS Settings
- Application credentials
- OAuth configuration
- Environment selection (sandbox/production)
- Webhook signature key

### Notification Settings
- SendGrid API configuration
- SMTP server settings
- Twilio SMS configuration
- From address and sender information

### AWS Settings
- Region configuration
- Access key and secret management
- KMS key configuration for encryption
- Service-specific settings

## Key Features

### Environment Management
- Automatic loading from .env files
- Environment variable precedence
- Type coercion and validation
- Default value management
- Case-insensitive variable names

### Validation
- Built-in validation for required configurations
- Custom validation methods
- Error reporting for missing configurations
- Type checking and coercion
- Range validation for numeric values

### Caching
- Cached settings instance for performance
- Single instantiation pattern
- Thread-safe access
- Lazy loading of settings
- Memory-efficient storage

### Security
- Secure handling of sensitive credentials
- Environment variable-based secrets
- No hardcoded credentials
- Encrypted storage for sensitive data
- Secure configuration loading

## Integration Points
- API layer for application configuration
- Database layer for connection settings
- Search layer for search service configuration
- Payment layer for payment service settings
- Notification layer for messaging services
- Task queue for background processing

## Best Practices
- Use environment variables for all configuration
- Never hardcode sensitive information
- Validate all required configurations at startup
- Use appropriate default values
- Document all configuration options
- Follow security best practices for credential handling

## Error Handling
- Comprehensive validation error reporting
- Graceful startup failure for invalid configurations
- Detailed error messages for debugging
- Configuration requirement checking
- Type validation error handling