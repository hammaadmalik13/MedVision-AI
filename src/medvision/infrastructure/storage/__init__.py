"""Local and S3 storage implementations."""

from pathlib import Path

import aiofiles
import boto3
from botocore.exceptions import ClientError

from medvision.config import get_settings
from medvision.domain.interfaces.storage import StorageService


class LocalStorageService(StorageService):
    def __init__(self, base_path: str | None = None) -> None:
        settings = get_settings()
        self.base_path = Path(base_path or settings.storage_local_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload(self, local_path: Path, remote_key: str) -> str:
        dest = self.base_path / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(local_path, "rb") as src:
            content = await src.read()
        async with aiofiles.open(dest, "wb") as dst:
            await dst.write(content)
        return str(dest)

    async def download(self, remote_key: str, local_path: Path) -> Path:
        src = self.base_path / remote_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(src, "rb") as f:
            content = await f.read()
        async with aiofiles.open(local_path, "wb") as f:
            await f.write(content)
        return local_path

    async def delete(self, remote_key: str) -> bool:
        path = self.base_path / remote_key
        if path.exists():
            path.unlink()
            return True
        return False

    async def get_url(self, remote_key: str, expires_in: int = 3600) -> str:
        return str(self.base_path / remote_key)

    async def exists(self, remote_key: str) -> bool:
        return (self.base_path / remote_key).exists()


class S3StorageService(StorageService):
    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_region,
        )

    async def upload(self, local_path: Path, remote_key: str) -> str:
        self.client.upload_file(str(local_path), self.bucket, remote_key)
        return f"s3://{self.bucket}/{remote_key}"

    async def download(self, remote_key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, remote_key, str(local_path))
        return local_path

    async def delete(self, remote_key: str) -> bool:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=remote_key)
            return True
        except ClientError:
            return False

    async def get_url(self, remote_key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": remote_key},
            ExpiresIn=expires_in,
        )

    async def exists(self, remote_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=remote_key)
            return True
        except ClientError:
            return False


def get_storage_service() -> StorageService:
    settings = get_settings()
    if settings.storage_backend == "s3":
        return S3StorageService()
    return LocalStorageService()
