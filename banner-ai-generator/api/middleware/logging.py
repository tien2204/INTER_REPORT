"""
Logging Middleware

Comprehensive request/response logging with performance metrics.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import json
from typing import Optional
from structlog import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/response logging middleware with performance tracking
    """
    
    def __init__(self, app, config: Optional[dict] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Logging settings
        self.enabled = self.config.get("enabled", True)
        self.log_requests = self.config.get("log_requests", True)
        self.log_responses = self.config.get("log_responses", True)
        self.log_headers = self.config.get("log_headers", False)
        self.log_body = self.config.get("log_body", False)
        self.log_performance = self.config.get("log_performance", True)
        
        # Performance thresholds
        self.slow_request_threshold = self.config.get("slow_request_threshold", 2.0)  # seconds
        self.very_slow_request_threshold = self.config.get("very_slow_request_threshold", 5.0)
        
        # Paths to exclude from logging
        self.exclude_paths = self.config.get("exclude_paths", [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        ])
        
        # Sensitive headers to redact
        self.sensitive_headers = self.config.get("sensitive_headers", [
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie"
        ])
        
        # Max body size to log (in bytes)
        self.max_body_size = self.config.get("max_body_size", 1024)
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response with performance metrics"""
        if not self.enabled or self._should_exclude_path(request.url.path):
            return await call_next(request)
        
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request)
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(
                "Request failed with exception",
                path=request.url.path,
                method=request.method,
                duration=duration,
                exception=str(e),
                request_id=getattr(request.state, "request_id", None),
                user_id=getattr(request.state, "user_id", None)
            )
            raise
        
        # Calculate performance metrics
        end_time = time.time()
        duration = end_time - start_time
        
        # Log response
        if self.log_responses:
            await self._log_response(request, response, duration)
        
        # Log performance metrics
        if self.log_performance:
            await self._log_performance(request, response, duration)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from logging"""
        return any(path.startswith(exclude_path) for exclude_path in self.exclude_paths)
    
    async def _log_request(self, request: Request):
        """Log incoming request details"""
        try:
            # FIXED: Remove 'event' key to avoid conflict with structlog
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params) if request.query_params else None,
                "request_id": getattr(request.state, "request_id", None),
                "user_id": getattr(request.state, "user_id", None),
                "user_role": getattr(request.state, "user_role", None),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length")
            }
            
            # Add headers if enabled
            if self.log_headers:
                log_data["headers"] = self._sanitize_headers(dict(request.headers))
            
            # Add body if enabled and present
            if self.log_body:
                body = await self._get_request_body(request)
                if body:
                    log_data["body"] = body
            
            # FIXED: Use message and separate parameters instead of **log_data with 'event' key
            logger.info("Incoming request", **log_data)
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
    
    async def _log_response(self, request: Request, response: Response, duration: float):
        """Log response details"""
        try:
            # FIXED: Remove 'event' key to avoid conflict with structlog
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": round(duration, 3),
                "request_id": getattr(request.state, "request_id", None),
                "user_id": getattr(request.state, "user_id", None),
                "response_size": response.headers.get("content-length")
            }
            
            # Add response headers if enabled
            if self.log_headers:
                log_data["response_headers"] = self._sanitize_headers(dict(response.headers))
            
            # Determine log level based on status code
            if response.status_code >= 500:
                logger.error("Response sent", **log_data)
            elif response.status_code >= 400:
                logger.warning("Response sent", **log_data)
            else:
                logger.info("Response sent", **log_data)
                
        except Exception as e:
            logger.error(f"Error logging response: {e}")
    
    async def _log_performance(self, request: Request, response: Response, duration: float):
        """Log performance metrics"""
        try:
            # Determine performance category
            if duration >= self.very_slow_request_threshold:
                performance_category = "very_slow"
                log_level = "warning"
            elif duration >= self.slow_request_threshold:
                performance_category = "slow"
                log_level = "warning"
            else:
                performance_category = "normal"
                log_level = "debug"
            
            # FIXED: Remove 'event' key to avoid conflict with structlog
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": round(duration, 3),
                "performance_category": performance_category,
                "request_id": getattr(request.state, "request_id", None),
                "user_id": getattr(request.state, "user_id", None)
            }
            
            # Add timing breakdown if available
            start_time = getattr(request.state, "start_time", None)
            if start_time:
                log_data["processing_time"] = round(time.time() - start_time, 3)
            
            # Log based on performance
            if log_level == "warning":
                logger.warning("Performance metric", **log_data)
            else:
                logger.debug("Performance metric", **log_data)
                
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def _sanitize_headers(self, headers: dict) -> dict:
        """Sanitize headers by redacting sensitive information"""
        try:
            sanitized = {}
            
            for key, value in headers.items():
                key_lower = key.lower()
                
                if key_lower in self.sensitive_headers:
                    # Redact sensitive headers
                    if key_lower == "authorization" and value.startswith("Bearer "):
                        sanitized[key] = f"Bearer {value[7:15]}..." if len(value) > 15 else "Bearer ***"
                    elif key_lower == "x-api-key":
                        sanitized[key] = f"{value[:8]}..." if len(value) > 8 else "***"
                    else:
                        sanitized[key] = "***"
                else:
                    sanitized[key] = value
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing headers: {e}")
            return {"error": "Failed to sanitize headers"}
    
    async def _get_request_body(self, request: Request) -> Optional[str]:
        """Get request body for logging"""
        try:
            # Only log small bodies
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_body_size:
                return f"<body too large: {content_length} bytes>"
            
            # Read body
            body = await request.body()
            
            if not body:
                return None
            
            if len(body) > self.max_body_size:
                return f"<body truncated: {len(body)} bytes>"
            
            # Try to decode as text
            try:
                return body.decode('utf-8')
            except UnicodeDecodeError:
                return f"<binary body: {len(body)} bytes>"
                
        except Exception as e:
            logger.error(f"Error getting request body: {e}")
            return "<error reading body>"
    
    async def log_custom_event(self, event_name: str, data: dict, request: Optional[Request] = None):
        """Log custom events with optional request context"""
        try:
            # FIXED: Don't use 'event' as a key in log_data
            log_data = {
                "event_name": event_name,  # Changed from 'event' to 'event_name'
                "timestamp": time.time(),
                **data
            }
            
            # Add request context if available
            if request:
                log_data.update({
                    "request_id": getattr(request.state, "request_id", None),
                    "user_id": getattr(request.state, "user_id", None),
                    "path": request.url.path,
                    "method": request.method
                })
            
            logger.info("Custom event", **log_data)
            
        except Exception as e:
            logger.error(f"Error logging custom event: {e}")
    
    def get_statistics(self) -> dict:
        """Get logging statistics"""
        return {
            "logging_enabled": self.enabled,
            "log_requests": self.log_requests,
            "log_responses": self.log_responses,
            "log_headers": self.log_headers,
            "log_body": self.log_body,
            "excluded_paths": self.exclude_paths,
            "slow_request_threshold": self.slow_request_threshold,
            "very_slow_request_threshold": self.very_slow_request_threshold,
            "max_body_size": self.max_body_size
        }
