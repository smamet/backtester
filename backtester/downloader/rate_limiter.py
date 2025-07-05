"""
Rate limiter for API requests to respect Binance limits.
"""

import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import deque
import logging


class RateLimiter:
    """
    Rate limiter to manage API requests and respect Binance limits.
    
    Features:
    - IP-based rate limiting (6000 requests per minute)
    - Weight-based limiting (each endpoint has different weights)
    - Automatic backoff on rate limit errors
    - Request tracking with sliding window
    """
    
    def __init__(self, 
                 requests_per_minute: int = 6000,
                 weight_per_minute: int = 6000,
                 buffer_ratio: float = 0.1,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            weight_per_minute: Maximum weight per minute
            buffer_ratio: Safety buffer (0.1 = 10% buffer)
            logger: Logger instance
        """
        self.requests_per_minute = requests_per_minute
        self.weight_per_minute = weight_per_minute
        self.buffer_ratio = buffer_ratio
        self.logger = logger or logging.getLogger(__name__)
        
        # Calculate effective limits with buffer
        self.effective_requests_per_minute = int(requests_per_minute * (1 - buffer_ratio))
        self.effective_weight_per_minute = int(weight_per_minute * (1 - buffer_ratio))
        
        # Request tracking
        self.request_times = deque()
        self.request_weights = deque()
        
        # Current usage tracking
        self.current_requests = 0
        self.current_weight = 0
        
        # Backoff state
        self.backoff_until = None
        self.backoff_duration = 1.0  # Start with 1 second
        self.max_backoff = 300.0  # Max 5 minutes
        
        self.logger.info(f"RateLimiter initialized: {self.effective_requests_per_minute} req/min, "
                        f"{self.effective_weight_per_minute} weight/min")
    
    def _cleanup_old_requests(self):
        """Remove requests older than 1 minute."""
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute ago
        
        # Remove old requests
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
            self.request_weights.popleft()
        
        # Update current counts
        self.current_requests = len(self.request_times)
        self.current_weight = sum(self.request_weights)
    
    def _calculate_wait_time(self, weight: int = 1) -> float:
        """Calculate how long to wait before next request."""
        self._cleanup_old_requests()
        
        # If we're in backoff, respect it
        if self.backoff_until:
            remaining_backoff = self.backoff_until - time.time()
            if remaining_backoff > 0:
                return remaining_backoff
            else:
                self.backoff_until = None
        
        # Check if we would exceed limits
        if (self.current_requests + 1 > self.effective_requests_per_minute or
            self.current_weight + weight > self.effective_weight_per_minute):
            
            # Calculate when the oldest request will expire
            if self.request_times:
                oldest_request_time = self.request_times[0]
                wait_time = 60 - (time.time() - oldest_request_time)
                return max(0, wait_time)
        
        return 0
    
    def wait_if_needed(self, weight: int = 1) -> float:
        """
        Wait if needed to respect rate limits.
        
        Args:
            weight: Weight of the request
            
        Returns:
            Time waited in seconds
        """
        wait_time = self._calculate_wait_time(weight)
        
        if wait_time > 0:
            self.logger.debug(f"Rate limit wait: {wait_time:.2f}s (weight: {weight})")
            time.sleep(wait_time)
        
        # Record this request
        current_time = time.time()
        self.request_times.append(current_time)
        self.request_weights.append(weight)
        
        self._cleanup_old_requests()
        
        return wait_time
    
    async def wait_if_needed_async(self, weight: int = 1) -> float:
        """
        Async version of wait_if_needed.
        
        Args:
            weight: Weight of the request
            
        Returns:
            Time waited in seconds
        """
        wait_time = self._calculate_wait_time(weight)
        
        if wait_time > 0:
            self.logger.debug(f"Rate limit wait (async): {wait_time:.2f}s (weight: {weight})")
            await asyncio.sleep(wait_time)
        
        # Record this request
        current_time = time.time()
        self.request_times.append(current_time)
        self.request_weights.append(weight)
        
        self._cleanup_old_requests()
        
        return wait_time
    
    def handle_rate_limit_error(self, retry_after: Optional[int] = None):
        """
        Handle rate limit error by setting backoff.
        
        Args:
            retry_after: Seconds to wait from API response
        """
        if retry_after:
            self.backoff_until = time.time() + retry_after
            self.logger.warning(f"Rate limit hit! Backing off for {retry_after}s")
        else:
            # Exponential backoff
            self.backoff_until = time.time() + self.backoff_duration
            self.logger.warning(f"Rate limit hit! Backing off for {self.backoff_duration}s")
            
            # Increase backoff duration for next time
            self.backoff_duration = min(self.backoff_duration * 2, self.max_backoff)
    
    def reset_backoff(self):
        """Reset backoff state after successful request."""
        self.backoff_until = None
        self.backoff_duration = 1.0
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        self._cleanup_old_requests()
        
        return {
            "current_requests": self.current_requests,
            "max_requests": self.effective_requests_per_minute,
            "current_weight": self.current_weight,
            "max_weight": self.effective_weight_per_minute,
            "requests_utilization": self.current_requests / self.effective_requests_per_minute,
            "weight_utilization": self.current_weight / self.effective_weight_per_minute,
            "backoff_active": self.backoff_until is not None,
            "backoff_remaining": max(0, self.backoff_until - time.time()) if self.backoff_until else 0
        }
    
    def can_make_request(self, weight: int = 1) -> bool:
        """
        Check if we can make a request without waiting.
        
        Args:
            weight: Weight of the request
            
        Returns:
            True if request can be made immediately
        """
        if self.backoff_until and time.time() < self.backoff_until:
            return False
        
        self._cleanup_old_requests()
        
        return (self.current_requests < self.effective_requests_per_minute and
                self.current_weight + weight <= self.effective_weight_per_minute)
    
    def estimate_wait_time(self, weight: int = 1) -> float:
        """
        Estimate wait time without actually waiting.
        
        Args:
            weight: Weight of the request
            
        Returns:
            Estimated wait time in seconds
        """
        return self._calculate_wait_time(weight)
    
    def optimal_batch_size(self, weight_per_request: int) -> int:
        """
        Calculate optimal batch size based on current limits.
        
        Args:
            weight_per_request: Weight per individual request
            
        Returns:
            Optimal number of requests that can be made in current window
        """
        self._cleanup_old_requests()
        
        # Calculate how many requests we can make based on weight
        weight_available = self.effective_weight_per_minute - self.current_weight
        requests_by_weight = weight_available // weight_per_request
        
        # Calculate how many requests we can make based on count
        requests_available = self.effective_requests_per_minute - self.current_requests
        
        # Return the minimum
        return max(0, min(requests_by_weight, requests_available))
    
    def log_stats(self):
        """Log current usage statistics."""
        stats = self.get_usage_stats()
        self.logger.info(
            f"Rate limiter stats: "
            f"Requests: {stats['current_requests']}/{stats['max_requests']} "
            f"({stats['requests_utilization']:.1%}), "
            f"Weight: {stats['current_weight']}/{stats['max_weight']} "
            f"({stats['weight_utilization']:.1%})"
        ) 