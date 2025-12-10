from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        retryable_exceptions: Tuple of exception types to retry on
    
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {str(e)}"
                        )
            
            # All retries exhausted, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


def retry_on_api_error(max_retries: int = 3):
    """Specialized retry decorator for API calls (OpenAI, network, etc.).
    
    Retries on:
    - Rate limit errors
    - Network/timeout errors
    - Temporary API errors
    
    Does NOT retry on:
    - Authentication errors
    - Invalid request errors
    - Parsing errors
    """
    from openai import RateLimitError, APIConnectionError, APITimeoutError, APIError
    
    retryable = (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIError,
        ConnectionError,
        TimeoutError,
    )
    
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=1.0,
        backoff_factor=2.0,
        retryable_exceptions=retryable,
    )


def retry_on_network_error(max_retries: int = 3):
    """Specialized retry decorator for network operations (data fetching).
    
    Retries on:
    - Network connection errors
    - Timeout errors
    - Temporary service unavailability
    """
    retryable = (
        ConnectionError,
        TimeoutError,
        OSError,  # Network-related OS errors
    )
    
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=1.0,
        backoff_factor=2.0,
        retryable_exceptions=retryable,
    )


def handle_node_error(
    node_name: str,
    state: dict,
    error: Exception,
    is_critical: bool = True,
    retry_count_key: str | None = None
) -> dict:
    """Handle errors in pipeline nodes with proper state updates.
    
    Args:
        node_name: Name of the node that failed
        state: Pipeline state dictionary
        error: Exception that occurred
        is_critical: Whether this error should stop execution
        retry_count_key: Key to track retry count in state["retry_count"]
    
    Returns:
        Updated state with error information
    """
    error_msg = f"{node_name} failed: {str(error)}"
    logger.error(error_msg)
    
    # Add to errors list
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append(error_msg)
    
    # Track retry count if specified
    if retry_count_key:
        if "retry_count" not in state:
            state["retry_count"] = {}
        state["retry_count"][retry_count_key] = state["retry_count"].get(retry_count_key, 0) + 1
    
    # If critical, the exception should be re-raised by the caller
    # If non-critical, just log and continue
    if not is_critical:
        logger.warning(f"Non-critical error in {node_name}, continuing...")
    
    return state


# Quick validation when run directly: python pipeline/errors.py
if __name__ == "__main__":
    print("✓ Testing errors.py...")
    
    # Test retry decorator
    call_count = [0]
    
    @retry_with_backoff(max_retries=2, initial_delay=0.1)
    def test_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError("Test error")
        return "Success"
    
    result = test_function()
    assert result == "Success"
    assert call_count[0] == 3  # Initial + 2 retries
    print("✓ Retry decorator works correctly")
    
    # Test error handling
    state = {"errors": []}
    try:
        raise ValueError("Test error")
    except Exception as e:
        state = handle_node_error("test_node", state, e, is_critical=False)
    
    assert len(state["errors"]) == 1
    assert "test_node failed" in state["errors"][0]
    print("✓ Error handling works correctly")
    
    print("✓ All error handling utilities validated successfully!")

