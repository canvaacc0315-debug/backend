# auth.py
import os
import json
from typing import Optional

import requests
from fastapi import Header, HTTPException, status

from jose import jwt

CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL")
CLERK_ISSUER = os.getenv("CLERK_ISSUER")  # optional
CLERK_AUDIENCE = os.getenv("CLERK_AUDIENCE")  # optional

# 🔐 DEV FLAG: set SKIP_AUTH=true in .env to bypass real JWT checks locally
SKIP_AUTH = os.getenv("SKIP_AUTH", "true").lower() == "true"


def _get_jwks():
    if not CLERK_JWKS_URL:
        raise RuntimeError("CLERK_JWKS_URL not set (check your .env)")
    resp = requests.get(CLERK_JWKS_URL)
    resp.raise_for_status()
    return resp.json()


def verify_clerk_token(token: str) -> str:
    """
    Verify Clerk JWT and return user_id. In dev you can bypass this
    by setting SKIP_AUTH=true in .env.
    """
    jwks = _get_jwks()
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing kid in token header",
        )

    key = None
    for k in jwks["keys"]:
        if k["kid"] == kid:
            key = k
            break

    if key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No matching JWK for token",
        )

    options = {"verify_aud": bool(CLERK_AUDIENCE)}
    decoded = jwt.decode(
        token,
        key,
        algorithms=[key["alg"]],
        audience=CLERK_AUDIENCE,
        issuer=CLERK_ISSUER,
        options=options,
    )

    # Clerk puts user id in "sub"
    return decoded.get("sub") or decoded.get("user_id")


async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Returns user_id. In dev (SKIP_AUTH=true) it will just return 'dev-user'
    even if the token is missing or invalid.
    """
    # ✅ DEV SHORTCUT – makes your 401 go away while building the UI
    if SKIP_AUTH:
        return "dev-user"

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    try:
        scheme, token = authorization.split(" ", 1)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )

    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization scheme must be Bearer",
        )

    try:
        user_id = verify_clerk_token(token)
    except Exception as e:
        print("Token verification failed:", repr(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return user_id