"""S3-compatible object storage helper (MinIO locally / S3 or R2 in prod).

Per CLAUDE.md: binary media never lives in Postgres — only object storage.
"""
from functools import lru_cache
from typing import BinaryIO

import boto3

from app.config import get_settings


@lru_cache
def get_s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
    )


def ensure_bucket() -> None:
    settings = get_settings()
    client = get_s3_client()
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if settings.s3_bucket not in existing:
        client.create_bucket(Bucket=settings.s3_bucket)


def upload_media(key: str, fileobj: BinaryIO, content_type: str | None = None) -> None:
    settings = get_settings()
    client = get_s3_client()
    extra_args = {"ContentType": content_type} if content_type else {}
    client.upload_fileobj(fileobj, settings.s3_bucket, key, ExtraArgs=extra_args)


def download_media(key: str) -> bytes:
    settings = get_settings()
    client = get_s3_client()
    obj = client.get_object(Bucket=settings.s3_bucket, Key=key)
    return obj["Body"].read()
