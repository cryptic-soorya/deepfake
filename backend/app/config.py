import os
from functools import lru_cache


class Settings:
    def __init__(self) -> None:
        self.database_url = os.environ["DATABASE_URL"]
        self.redis_url = os.environ["REDIS_URL"]
        self.s3_endpoint = os.environ["S3_ENDPOINT"]
        self.s3_access_key = os.environ["S3_ACCESS_KEY"]
        self.s3_secret_key = os.environ["S3_SECRET_KEY"]
        self.s3_bucket = os.environ["S3_BUCKET"]
        self.gemini_api_key = os.environ["GEMINI_API_KEY"]
        self.jwt_secret = os.environ["JWT_SECRET"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
