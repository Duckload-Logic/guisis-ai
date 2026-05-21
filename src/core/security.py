import logging
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.core.config import settings

logger = logging.getLogger(__name__)

# Enforce strict 80-character limit per line in this file
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def validate_api_key(
    api_key_value: str = Security(api_key_header)
):
    if not settings.allowed_api_key:
        logger.warning(
            "[Security] ALLOWED_API_KEY not configured. "
            "Bypassing validation check."
        )
        return None

    if api_key_value != settings.allowed_api_key:
        logger.warning(
            "[Security] Invalid API key attempt: "
            f"{api_key_value[:4]}..." if api_key_value else "None"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid or missing X-API-Key header"
        )

    return api_key_value
