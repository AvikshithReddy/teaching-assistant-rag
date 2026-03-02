import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

_SECRET = os.getenv("JWT_SECRET") or secrets.token_hex(32)
_ALGO = "HS256"
_EXPIRY_DAYS = 7


def create_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_EXPIRY_DAYS)
    payload = {"sub": user_id, "role": role, "exp": expire}
    return jwt.encode(payload, _SECRET, algorithm=_ALGO)


def verify_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, _SECRET, algorithms=[_ALGO])
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc
