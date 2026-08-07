"""Audit logging for every endpoint that touches media or biometric data.

Per CLAUDE.md this is not optional: every such endpoint must write an
audit log entry in the same PR that adds the endpoint.
"""

from datetime import datetime, timezone


async def write_audit_log(*, user_id: str, action: str, resource_id: str, metadata: dict | None = None) -> None:
    raise NotImplementedError
