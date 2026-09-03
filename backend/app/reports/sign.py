"""Cryptographic signing of generated forensic reports so they can be verified as untampered.

HMAC-SHA256 over a canonical payload, keyed with the same JWT_SECRET the
rest of the app already trusts as a server-held secret -- no new secret to
provision. Verification just means recomputing the HMAC with that same key
and comparing digests, so anyone downstream of us can prove a report wasn't
altered after we generated it as long as they hold (or are given) the key.
"""
import hashlib
import hmac

from app.config import get_settings


def sign_report(payload: bytes) -> bytes:
    """Return the hex-encoded HMAC-SHA256 signature of `payload`."""
    settings = get_settings()
    digest = hmac.new(settings.jwt_secret.encode(), payload, hashlib.sha256).hexdigest()
    return digest.encode()


def verify_report(payload: bytes, signature: bytes) -> bool:
    expected = sign_report(payload)
    return hmac.compare_digest(expected, signature)
