"""Sentinel API key authentication."""

import os
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Check API key header against API_KEY env var. Skips auth if env var is empty."""
    expected = os.environ.get("API_KEY", "")
    if not expected:
        return
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
