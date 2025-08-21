"""
Rate Limiting Middleware

Implements rate limiting to prevent API abuse and ensure fair usage.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from structlog import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with sliding window algorithm
    """
    
    def __init__(self, app, config: Optional[dict] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Rate limiting settings
        self.enabled = self.config.get("enabled", True)
        self.requests_per_minute = self.config.get("requests_per_minute", 60)
        self.requests_per_hour = self.config.get("requests_per_hour", 1000)
        self.burst_limit = self.config.get("burst_limit", 10)  # Short burst allowance
        
        # Different limits for different user types
        self.limits_by_role = self.config.get("limits_by_role", {
            "admin": {"per_minute": 300, "per_hour": 10000},
            "api_user": {"per_minute": 120, "per_hour": 5000},
            "user": {"per_minute": 60, "per_hour": 1000},
            "guest": {"per_minute": 20, "per_hour": 200}
        })
        
        # Endpoints with special limits
        self.endpoint_limits = self.config.get("endpoint_limits", {
            "/api/v1/designs/generate": {"per_minute": 10, "per_hour": 100},
            "/api/v1/assets/upload": {"per_minute": 30, "per_hour": 500}
        })
        
        # Storage for rate limiting data
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.burst_counts: Dict[str, deque] = defaultdict(deque)
        
        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to incoming requests"""
        try:
            if not self.enabled:
                return await call_next(request)
            
            # Get client identifier
            client_id = self._get_client_id(request)
            
            # Get user role for role-based limits
            user_role = getattr(request.state, "user_role", "guest")
            
            # Check rate limits
            limit_result = await self._check_rate_limits(
                client_id, request.url.path, user_role
            )
            
            if not limit_result["allowed"]:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": 429,
                            "message": "Rate limit exceeded",
                            "type": "rate_limit_error",
                            "details": limit_result["details"]
                        },
                        "request_id": getattr(request.state, "request_id", None),
                        "retry_after": limit_result["retry_after"]
                    },
                    headers={
                        "Retry-After": str(limit_result["retry_after"]),
                        "X-RateLimit-Limit": str(limit_result["limit"]),
                        "X-RateLimit-Remaining": str(limit_result["remaining"]),
                        "X-RateLimit-Reset": str(limit_result["reset_time"])
                    }
                )
            
            # Record the request
            await self._record_request(client_id)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(limit_result["limit"])
            response.headers["X-RateLimit-Remaining"] = str(limit_result["remaining"])
            response.headers["X-RateLimit-Reset"] = str(limit_result["reset_time"])
            
            return response
            
        except Exception as e:
            logger.error(f"Error in rate limiting middleware: {e}")
            # Don't block requests on rate limiting errors
            return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Try to get user ID first
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limits(self, client_id: str, endpoint: str, user_role: str) -> dict:
        """Check if request is within rate limits"""
        try:
            current_time = time.time()
            
            # Get applicable limits
            limits = self._get_limits_for_client(endpoint, user_role)
            
            # Check minute limit
            minute_window_start = current_time - 60
            minute_requests = self._count_requests_in_window(
                client_id, minute_window_start, current_time
            )
            
            # Check hour limit
            hour_window_start = current_time - 3600
            hour_requests = self._count_requests_in_window(
                client_id, hour_window_start, current_time
            )
            
            # Check burst limit (last 10 seconds)
            burst_window_start = current_time - 10
            burst_requests = self._count_requests_in_window(
                client_id, burst_window_start, current_time
            )
            
            # Determine if request is allowed
            if burst_requests >= self.burst_limit:
                return {
                    "allowed": False,
                    "limit": self.burst_limit,
                    "remaining": 0,
                    "retry_after": 10,
                    "reset_time": int(current_time + 10),
                    "details": "Burst limit exceeded"
                }
            
            if minute_requests >= limits["per_minute"]:
                return {
                    "allowed": False,
                    "limit": limits["per_minute"],
                    "remaining": 0,
                    "retry_after": 60,
                    "reset_time": int(current_time + 60),
                    "details": "Per-minute limit exceeded"
                }
            
            if hour_requests >= limits["per_hour"]:
                return {
                    "allowed": False,
                    "limit": limits["per_hour"], 
                    "remaining": 0,
                    "retry_after": 3600,
                    "reset_time": int(current_time + 3600),
                    "details": "Per-hour limit exceeded"
                }
            
            # Request is allowed
            return {
                "allowed": True,
                "limit": limits["per_minute"],
                "remaining": limits["per_minute"] - minute_requests - 1,
                "retry_after": 0,
                "reset_time": int(current_time + 60),
                "details": "Within limits"
            }
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            # Allow request on error
            return {
                "allowed": True,
                "limit": self.requests_per_minute,
                "remaining": self.requests_per_minute,
                "retry_after": 0,
                "reset_time": int(time.time() + 60),
                "details": "Rate limit check failed, allowing request"
            }
    
    def _get_limits_for_client(self, endpoint: str, user_role: str) -> dict:
        """Get rate limits for specific client and endpoint"""
        # Start with default limits
        limits = {
            "per_minute": self.requests_per_minute,
            "per_hour": self.requests_per_hour
        }
        
        # Apply role-based limits
        if user_role in self.limits_by_role:
            role_limits = self.limits_by_role[user_role]
            limits.update(role_limits)
        
        # Apply endpoint-specific limits (most restrictive)
        for endpoint_pattern, endpoint_limits in self.endpoint_limits.items():
            if endpoint.startswith(endpoint_pattern):
                # Use the most restrictive limits
                limits["per_minute"] = min(limits["per_minute"], endpoint_limits.get("per_minute", limits["per_minute"]))
                limits["per_hour"] = min(limits["per_hour"], endpoint_limits.get("per_hour", limits["per_hour"]))
                break
        
        return limits
    
    def _count_requests_in_window(self, client_id: str, window_start: float, window_end: float) -> int:
        """Count requests for client in time window"""
        try:
            request_queue = self.request_counts[client_id]
            
            # Remove old requests
            while request_queue and request_queue[0] < window_start:
                request_queue.popleft()
            
            # Count requests in window
            count = 0
            for request_time in request_queue:
                if window_start <= request_time <= window_end:
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error counting requests: {e}")
            return 0
    
    async def _record_request(self, client_id: str):
        """Record a request for rate limiting"""
        try:
            current_time = time.time()
            
            # Add to request queue
            self.request_counts[client_id].append(current_time)
            
            # Limit queue size to prevent memory issues
            max_queue_size = 1000
            if len(self.request_counts[client_id]) > max_queue_size:
                # Remove oldest requests
                for _ in range(100):
                    if self.request_counts[client_id]:
                        self.request_counts[client_id].popleft()
            
        except Exception as e:
            logger.error(f"Error recording request: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    await self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup())
    
    async def _cleanup_old_data(self):
        """Clean up old rate limiting data"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600  # Keep 1 hour of data
            
            clients_to_remove = []
            
            for client_id in list(self.request_counts.keys()):
                request_queue = self.request_counts[client_id]
                
                # Remove old requests
                while request_queue and request_queue[0] < cutoff_time:
                    request_queue.popleft()
                
                # Remove client if no recent requests
                if not request_queue:
                    clients_to_remove.append(client_id)
            
            # Remove empty clients
            for client_id in clients_to_remove:
                del self.request_counts[client_id]
                if client_id in self.burst_counts:
                    del self.burst_counts[client_id]
            
            logger.debug(f"Cleaned up rate limiting data for {len(clients_to_remove)} inactive clients")
            
        except Exception as e:
            logger.error(f"Error cleaning up rate limiting data: {e}")
    
    def get_client_stats(self, client_id: str) -> dict:
        """Get statistics for a specific client"""
        try:
            current_time = time.time()
            
            minute_count = self._count_requests_in_window(
                client_id, current_time - 60, current_time
            )
            hour_count = self._count_requests_in_window(
                client_id, current_time - 3600, current_time
            )
            
            return {
                "client_id": client_id,
                "requests_last_minute": minute_count,
                "requests_last_hour": hour_count,
                "total_tracked_requests": len(self.request_counts.get(client_id, [])),
                "last_request_time": self.request_counts[client_id][-1] if client_id in self.request_counts and self.request_counts[client_id] else None
            }
            
        except Exception as e:
            logger.error(f"Error getting client stats: {e}")
            return {"error": str(e)}
    
    def get_global_stats(self) -> dict:
        """Get global rate limiting statistics"""
        try:
            current_time = time.time()
            
            total_clients = len(self.request_counts)
            total_requests_minute = 0
            total_requests_hour = 0
            
            for client_id in self.request_counts:
                total_requests_minute += self._count_requests_in_window(
                    client_id, current_time - 60, current_time
                )
                total_requests_hour += self._count_requests_in_window(
                    client_id, current_time - 3600, current_time
                )
            
            return {
                "total_active_clients": total_clients,
                "total_requests_last_minute": total_requests_minute,
                "total_requests_last_hour": total_requests_hour,
                "average_requests_per_client_minute": total_requests_minute / max(total_clients, 1),
                "rate_limiting_enabled": self.enabled,
                "global_limits": {
                    "requests_per_minute": self.requests_per_minute,
                    "requests_per_hour": self.requests_per_hour,
                    "burst_limit": self.burst_limit
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup when middleware is destroyed"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
