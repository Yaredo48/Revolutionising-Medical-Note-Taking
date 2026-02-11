"""
Rate limiting middleware for AI Service Platform.
Implements Redis-based rate limiting per tenant.
"""

import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

import redis.asyncio as redis
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

import sys
sys.path.insert(0, '/home/yared/Documents/GenAIProject/Revolutionising-Medical-Note-Taking/ai_service_platform')

from src.core.config import settings


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: int
    limit: int


class RateLimiter:
    """Redis-based rate limiter."""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis = redis_client
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if not self.redis:
            self.redis = redis.from_url(settings.redis.url)
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    async def check_rate_limit(
        self,
        tenant_id: int,
        limit: int = 100,
        window: int = 60,  # seconds
        key_prefix: str = "ratelimit"
    ) -> RateLimitResult:
        """
        Check if a request is within rate limits.
        
        Uses sliding window algorithm with Redis.
        """
        current_time = int(time.time())
        window_start = current_time - window
        
        key = f"{key_prefix}:{tenant_id}:{current_time // window}"
        
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
        pipe.zcard(key)  # Count current requests
        pipe.zadd(key, {str(current_time): current_time})  # Add current request
        pipe.expire(key, window)  # Set expiry
        
        results = await pipe.execute()
        current_count = results[1]
        
        remaining = max(0, limit - current_count - 1)
        allowed = current_count < limit
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=((current_time // window) + 1) * window,
            limit=limit
        )
    
    async def check_tokens_rate_limit(
        self,
        tenant_id: int,
        tokens: int,
        limit: int = 10000,
        window: int = 60,
        key_prefix: str = "tokenratelimit"
    ) -> RateLimitResult:
        """Check token-based rate limit."""
        current_time = int(time.time())
        window_start = current_time - window
        
        key = f"{key_prefix}:{tenant_id}:{current_time // window}"
        
        pipe = self.redis.pipeline()
        pipe.get(key)  # Get current token count
        pipe.execute()
        
        current_tokens = int(await self.redis.get(key) or 0)
        new_tokens = current_tokens + tokens
        
        allowed = new_tokens <= limit
        
        if allowed:
            await self.redis.setex(key, window, str(new_tokens))
        
        remaining = max(0, limit - new_tokens)
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=((current_time // window) + 1) * window,
            limit=limit
        )


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces rate limits on all requests."""
    
    def __init__(self, app, prefix: str = "/api"):
        super().__init__(app)
        self.prefix = prefix
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Extract tenant ID from path or headers
        tenant_id = self._extract_tenant_id(request)
        
        if not tenant_id:
            return await call_next(request)
        
        # Check rate limit
        rate_limiter = await get_rate_limiter()
        result = await rate_limiter.check_rate_limit(
            tenant_id=tenant_id,
            limit=settings.rate_limit.requests_per_minute,
            window=60
        )
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_at)
        
        # Return 429 if rate limited
        if not result.allowed:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": result.reset_at - int(time.time())
                },
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(result.reset_at),
                    "Retry-After": str(result.reset_at - int(time.time()))
                }
            )
        
        return response
    
    def _extract_tenant_id(self, request: Request) -> Optional[int]:
        """Extract tenant ID from request."""
        # Try to get from path
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "api":
            try:
                return int(path_parts[1])
            except ValueError:
                pass
        
        # Try to get from header
        return request.headers.get("X-Tenant-ID")


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)
