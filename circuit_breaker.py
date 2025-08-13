
import time
import logging
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self):
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker moving to CLOSED state - service recovered")
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN - {self.failure_count} failures detected")
    
    @property
    def is_closed(self):
        return self.state == CircuitState.CLOSED
    
    @property  
    def is_open(self):
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self):
        return self.state == CircuitState.HALF_OPEN

class APICircuitBreakerManager:
    """Manages circuit breakers for different API endpoints."""
    
    def __init__(self):
        self.breakers = {
            'binance_market_data': CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'binance_trading': CircuitBreaker(failure_threshold=2, recovery_timeout=60),
            'google_sheets': CircuitBreaker(failure_threshold=5, recovery_timeout=120),
            'telegram': CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'firebase': CircuitBreaker(failure_threshold=4, recovery_timeout=60)
        }
    
    def get_breaker(self, service_name):
        """Get circuit breaker for specific service."""
        return self.breakers.get(service_name)
    
    def get_status(self):
        """Get status of all circuit breakers."""
        status = {}
        for name, breaker in self.breakers.items():
            status[name] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'is_available': breaker.is_closed or breaker.is_half_open
            }
        return status
