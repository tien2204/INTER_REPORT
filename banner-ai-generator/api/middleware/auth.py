"""
Authentication Middleware

Handles API authentication and authorization.
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
from typing import Optional
from structlog import get_logger

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication and authorization middleware
    """
    
    def __init__(self, app, config: Optional[dict] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Authentication settings
        self.auth_enabled = self.config.get("auth_enabled", False)  # Disabled by default for demo
        self.api_key_header = self.config.get("api_key_header", "X-API-Key")
        self.bearer_token_header = self.config.get("bearer_token_header", "Authorization")
        
        # Valid API keys (in production, load from secure storage)
        self.valid_api_keys = self.config.get("valid_api_keys", [
            "demo_api_key_12345",
            "test_api_key_67890"
        ])
        
        # Public endpoints that don't require auth
        self.public_endpoints = self.config.get("public_endpoints", [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        ])
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for incoming requests"""
        try:
            # Generate request ID for tracking
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            # Add request timestamp
            request.state.start_time = time.time()
            
            # Skip auth for public endpoints
            if self._is_public_endpoint(request.url.path):
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
            
            # Skip auth if disabled
            if not self.auth_enabled:
                # Add mock user for demo purposes
                request.state.user_id = "demo_user"
                request.state.user_role = "admin"
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response
            
            # Perform authentication
            auth_result = await self._authenticate_request(request)
            
            if not auth_result["valid"]:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": 401,
                            "message": auth_result["error"],
                            "type": "authentication_error"
                        },
                        "request_id": request_id
                    }
                )
            
            # Add user information to request state
            request.state.user_id = auth_result["user_id"]
            request.state.user_role = auth_result.get("role", "user")
            request.state.auth_method = auth_result["method"]
            
            # Process request
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            logger.error(f"Error in auth middleware: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Authentication service error",
                        "type": "auth_service_error"
                    },
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            )
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        return any(
            path.startswith(public_path) 
            for public_path in self.public_endpoints
        )
    
    async def _authenticate_request(self, request: Request) -> dict:
        """Authenticate the request"""
        try:
            # Try API key authentication
            api_key = request.headers.get(self.api_key_header)
            if api_key:
                return await self._authenticate_api_key(api_key)
            
            # Try Bearer token authentication
            auth_header = request.headers.get(self.bearer_token_header)
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]  # Remove "Bearer " prefix
                return await self._authenticate_bearer_token(token)
            
            # No authentication provided
            return {
                "valid": False,
                "error": "No authentication provided"
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                "valid": False,
                "error": "Authentication failed"
            }
    
    async def _authenticate_api_key(self, api_key: str) -> dict:
        """Authenticate using API key"""
        try:
            if api_key in self.valid_api_keys:
                # In production, look up user info from database
                return {
                    "valid": True,
                    "user_id": f"api_user_{api_key[:8]}",
                    "role": "api_user",
                    "method": "api_key"
                }
            else:
                return {
                    "valid": False,
                    "error": "Invalid API key"
                }
                
        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            return {
                "valid": False,
                "error": "API key authentication failed"
            }
    
    async def _authenticate_bearer_token(self, token: str) -> dict:
        """Authenticate using Bearer token (JWT)"""
        try:
            # TODO: Implement JWT token validation
            # For now, accept any token for demo purposes
            
            if len(token) > 10:  # Basic validation
                return {
                    "valid": True,
                    "user_id": f"jwt_user_{token[:8]}",
                    "role": "user",
                    "method": "bearer_token"
                }
            else:
                return {
                    "valid": False,
                    "error": "Invalid bearer token"
                }
                
        except Exception as e:
            logger.error(f"Bearer token authentication error: {e}")
            return {
                "valid": False,
                "error": "Bearer token authentication failed"
            }
    
    def _check_authorization(self, user_role: str, required_role: str) -> bool:
        """Check if user has required authorization"""
        role_hierarchy = {
            "admin": 3,
            "api_user": 2,
            "user": 1,
            "guest": 0
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 1)
        
        return user_level >= required_level
