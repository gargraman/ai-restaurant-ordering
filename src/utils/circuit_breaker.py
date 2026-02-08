"""Circuit breaker implementation for external API calls."""

import asyncio
import time
from enum import Enum
from typing import Callable, TypeVar, Awaitable, Any
import structlog

logger = structlog.get_logger()

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tripped, requests blocked
    HALF_OPEN = "half_open"  # Testing if failure condition is resolved


class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        
        # Lock to prevent race conditions
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    def _should_trip(self) -> bool:
        """Check if circuit should trip based on failure count."""
        return self._failure_count >= self.failure_threshold
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    # Attempt to transition to half-open state
                    self._state = CircuitState.HALF_OPEN
                    logger.info("circuit_breaker_half_open", 
                               failure_count=self._failure_count)
                else:
                    # Still in open state, raise exception
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker is OPEN. Last failure: {self._last_failure_time}"
                    )
        
        try:
            result = await func()
            
            # Success - reset failure count and close circuit if needed
            async with self._lock:
                self._failure_count = 0
                if self._state != CircuitState.CLOSED:
                    self._state = CircuitState.CLOSED
                    logger.info("circuit_breaker_closed", 
                               state_before=self._state.name)
            
            return result
            
        except self.expected_exception as e:
            # Failure - increment counter and possibly trip circuit
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if self._should_trip():
                    self._state = CircuitState.OPEN
                    logger.warning("circuit_breaker_opened",
                                  failure_count=self._failure_count,
                                  last_error=str(e))
                
            raise


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open and call is rejected."""
    pass


# Global circuit breakers for different services
_openai_api_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,  # 1 minute
    expected_exception=(Exception,)  # Catch all exceptions for API calls
)


def get_openai_circuit_breaker() -> CircuitBreaker:
    """Get the circuit breaker for OpenAI API calls."""
    return _openai_api_circuit_breaker