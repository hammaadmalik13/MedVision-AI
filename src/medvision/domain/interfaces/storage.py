"""Storage service interface."""

from abc import ABC, abstractmethod
from pathlib import Path


class StorageService(ABC):
    @abstractmethod
    async def upload(self, local_path: Path, remote_key: str) -> str: ...

    @abstractmethod
    async def download(self, remote_key: str, local_path: Path) -> Path: ...

    @abstractmethod
    async def delete(self, remote_key: str) -> bool: ...

    @abstractmethod
    async def get_url(self, remote_key: str, expires_in: int = 3600) -> str: ...

    @abstractmethod
    async def exists(self, remote_key: str) -> bool: ...
