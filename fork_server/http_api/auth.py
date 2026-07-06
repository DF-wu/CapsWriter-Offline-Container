# coding: utf-8
"""Dependency-light Bearer token helpers for the HTTP API."""

from __future__ import annotations

import hmac
from typing import Optional


def auth_enabled(api_key: Optional[str]) -> bool:
    return bool((api_key or "").strip())


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.casefold() != "bearer" or not token:
        return None
    return token


def bearer_token_matches(authorization: Optional[str], api_key: Optional[str]) -> bool:
    expected = (api_key or "").strip()
    if not expected:
        return True
    token = extract_bearer_token(authorization)
    if token is None:
        return False
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))
