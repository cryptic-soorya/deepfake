"""initial scan tables

Revision ID: 0001
Revises:
Create Date: 2026-08-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scans",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("media_key", sa.String(length=512), nullable=False),
        sa.Column("media_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("fused_score", sa.Float(), nullable=True),
        sa.Column("fused_verdict", sa.String(length=32), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "scan_results",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("scan_id", sa.String(length=36), sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("raw", sa.JSON(), nullable=True),
        sa.Column("result_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_scan_results_scan_id", "scan_results", ["scan_id"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("resource_id", sa.String(length=64), nullable=False),
        sa.Column("log_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_logs_resource_id", "audit_logs", ["resource_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_logs_resource_id", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_scan_results_scan_id", table_name="scan_results")
    op.drop_table("scan_results")
    op.drop_table("scans")
