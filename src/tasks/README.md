# Task Queue Documentation

## Overview
The task queue module provides Celery-based background task processing for async order processing. It handles order routing to POS systems with retry mechanisms, dead letter queue handling, and scheduled tasks for menu catalog synchronization.

## Architecture
- **Celery Application**: Core Celery configuration with Redis broker/backend
- **Order Tasks**: Order routing tasks with exponential backoff retry
- **Dead Letter Queue**: Handling for failed orders
- **Scheduled Tasks**: Periodic tasks via Celery Beat

## Key Components

### Celery Application (`celery_app.py`)
- Celery 5.x configuration with Redis as message broker and result backend
- Task routes for priority queues
- Serialization and timezone settings
- Task acknowledgment and visibility timeout
- Broker connection retry configuration

### Order Tasks (`order_tasks.py`)
- Order routing to POS systems with retry mechanisms
- Exponential backoff retry for temporary failures
- Immediate failure handling for permanent errors
- State machine transitions on success/failure
- Dead letter queue for orders exceeding max retries
- Menu catalog synchronization tasks

### Dead Letter Queue (`dlq.py`)
- Processing of orders that exceeded max retries
- Notification mechanisms for support and restaurants
- Manual retry capabilities
- Failure analysis and categorization
- Escalation procedures for persistent failures

### Beat Schedule (`beat_schedule.py`)
- Periodic tasks for menu catalog synchronization
- Stale order cleanup tasks
- Token refresh for POS connections
- Health check tasks
- Scheduled maintenance operations

## Key Features

### Retry Mechanisms
- Exponential backoff for temporary failures
- Configurable retry limits
- Different handling for temporary vs permanent errors
- Context preservation across retries
- Retry counting and tracking

### Order Processing
- Async order routing to POS systems
- State machine integration for status updates
- Error handling and recovery
- Dead letter queue for persistent failures
- Order tracking and monitoring

### Scheduled Operations
- Periodic menu catalog synchronization
- Regular cleanup of stale data
- Token refresh for external services
- System health monitoring
- Maintenance task scheduling

### Dead Letter Queue
- Capture of consistently failing tasks
- Manual intervention capabilities
- Failure analysis and reporting
- Escalation procedures
- Recovery mechanisms

## Integration Points
- Order management for POS routing
- POS systems for order injection
- Database layer for state updates
- Notification system for failure alerts
- Monitoring for task performance

## Error Handling
- Comprehensive exception handling
- Temporary vs permanent error classification
- Retry logic with exponential backoff
- Dead letter queue for persistent failures
- Detailed error logging and monitoring

## Security
- Secure task execution environment
- Input validation for task parameters
- Access controls for task execution
- Encrypted communication with broker

## Performance
- Configurable worker pools
- Priority queue support
- Resource optimization for task execution
- Connection pooling for broker communication
- Efficient serialization of task data