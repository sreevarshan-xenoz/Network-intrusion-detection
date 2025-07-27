"""
Authentication and security utilities for the API.
"""
import hashlib
import secrets
import time
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


class APIKeyManager:
    """Simple API key management for basic authentication."""
    
    def __init__(self):
        # In production, these should be stored securely (database, env vars, etc.)
        self._api_keys = {
            "demo_key_123": {
                "name": "demo_user",
                "permissions": ["predict", "batch_predict", "model_info"],
                "rate_limit": 1000,  # requests per hour
                "created_at": time.time()
            }
        }
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        return self._api_keys.get(api_key)
    
    def check_rate_limit(self, api_key: str, endpoint: str) -> bool:
        """Check if request is within rate limits."""
        if api_key not in self._api_keys:
            return False
        
        current_time = time.time()
        hour_window = int(current_time // 3600)
        
        if api_key not in self._rate_limits:
            self._rate_limits[api_key] = {}
        
        key = f"{endpoint}_{hour_window}"
        if key not in self._rate_limits[api_key]:
            self._rate_limits[api_key][key] = 0
        
        limit = self._api_keys[api_key]["rate_limit"]
        current_count = self._rate_limits[api_key][key]
        
        if current_count >= limit:
            return False
        
        self._rate_limits[api_key][key] += 1
        return True
    
    def generate_api_key(self, name: str, permissions: list, rate_limit: int = 1000) -> str:
        """Generate a new API key."""
        api_key = secrets.token_urlsafe(32)
        self._api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "created_at": time.time()
        }
        return api_key


# Global API key manager instance
api_key_manager = APIKeyManager()

# Security scheme
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Dependency to get current authenticated user."""
    api_key = credentials.credentials
    user_info = api_key_manager.validate_api_key(api_key)
    
    if not user_info:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"api_key": api_key, **user_info}


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(user: dict = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return user
    return permission_checker


def check_rate_limit(endpoint: str):
    """Decorator to check rate limits."""
    def rate_limit_checker(user: dict = Depends(get_current_user)):
        api_key = user["api_key"]
        if not api_key_manager.check_rate_limit(api_key, endpoint):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        return user
    return rate_limit_checker


def sanitize_input(data: str, max_length: int = 1000) -> str:
    """Basic input sanitization."""
    if len(data) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long. Maximum length is {max_length} characters."
        )
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
    for char in dangerous_chars:
        data = data.replace(char, '')
    
    return data.strip()