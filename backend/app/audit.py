"""Audit logging for every endpoint that touches media or biometric data.

Per CLAUDE.md this is not optional: every such endpoint must write an
audit log entry in the same PR that adds the endpoint.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import AuditLog


async def write_audit_log(
    session: AsyncSession,
    *,
    user_id: str | None,
    action: str,
    resource_id: str,
    metadata: dict | None = None,
) -> None:
    """Stage an audit log row on `session`. Caller controls the commit boundary
    so this can share a transaction with the write it's auditing."""
    session.add(
        AuditLog(
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            log_metadata=metadata,
        )
    )
    await session.flush()
