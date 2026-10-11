"""Azure Blob Storage: `az://<account>/<container>/<prefix>`.

The same tree, in a container, read with the
deployment's own identity. `azure-storage-blob` and `azure-identity` are the
`[azure]` extra and are imported HERE, when this backend is first used, so the
base install pulls in no cloud SDK.

**Credentials, in order:**

* `AZURE_STORAGE_CONNECTION_STRING`, when set - an account key or a SAS, and the
  way an emulator (Azurite) is reached. Its endpoint replaces the account in
  the URL; the container and prefix still come from the URL.
* otherwise `DefaultAzureCredential` against
  `https://<account>.blob.core.windows.net` - a managed identity in
  Azure, `az login` on a laptop.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from datasets_client.backends._blob import AlreadyExists, BlobBackend, Sink
from datasets_client.errors import BackendUnreachable, ConfigError, DatasetUnavailable


def parse(url: str) -> tuple[str, str, str]:
    """`az://account/container/prefix` -> (account, container, prefix/)."""
    parsed = urlparse(url)
    account = parsed.netloc
    parts = [p for p in parsed.path.split("/") if p]
    if not account or not parts:
        raise ConfigError(f"{url}: expected az://<account>/<container>[/<prefix>]")
    prefix = "/".join(parts[1:])
    return account, parts[0], (prefix + "/") if prefix else ""


class AzureStore:
    """`BlobStore` on one container, under one prefix."""

    def __init__(self, container_client, prefix: str) -> None:
        self.container = container_client
        self.prefix = prefix
        from azure.core.exceptions import (
            AzureError,
            ResourceExistsError,
            ResourceNotFoundError,
        )
        self._azure_error = AzureError
        self._exists = ResourceExistsError
        self._missing = ResourceNotFoundError

    def _unreachable(self, what: str, exc: Exception) -> BackendUnreachable:
        return BackendUnreachable(f"Azure Blob: could not {what}: {exc}")

    def children(self, prefix: str) -> list[str]:
        full = self.prefix + prefix
        try:
            out = []
            for item in self.container.walk_blobs(name_starts_with=full, delimiter="/"):
                name = item.name[len(full):]
                if name.endswith("/"):
                    out.append(name[:-1])
            return out
        except self._missing as exc:
            raise self._unreachable(f"list {full} (no such container)", exc) from exc
        except self._azure_error as exc:
            raise self._unreachable(f"list {full}", exc) from exc

    def keys(self, prefix: str) -> Iterator[str]:
        full = self.prefix + prefix
        try:
            for blob in self.container.list_blobs(name_starts_with=full):
                yield blob.name[len(self.prefix):]
        except self._azure_error as exc:
            raise self._unreachable(f"list {full}", exc) from exc

    def read(self, key: str) -> bytes | None:
        try:
            return self.container.download_blob(self.prefix + key).readall()
        except self._missing:
            return None
        except self._azure_error as exc:
            raise self._unreachable(f"read {key}", exc) from exc

    def download(self, key: str, sink: Sink) -> None:
        try:
            for chunk in self.container.download_blob(self.prefix + key).chunks():
                sink(chunk)
        except self._missing:
            raise DatasetUnavailable(f"{key} is listed but not in the container",
                                     status=404) from None
        except self._azure_error as exc:
            raise self._unreachable(f"read {key}", exc) from exc

    def put(self, key: str, source: Path | bytes, *, create_only: bool = False) -> None:
        try:
            if isinstance(source, bytes):
                self.container.upload_blob(self.prefix + key, source,
                                           overwrite=not create_only)
            else:
                with open(source, "rb") as data:
                    self.container.upload_blob(
                        self.prefix + key, data, length=os.path.getsize(source),
                        overwrite=not create_only, max_concurrency=4)
        except self._exists:
            raise AlreadyExists(key) from None
        except self._azure_error as exc:
            raise self._unreachable(f"write {key}", exc) from exc


def backend(url: str) -> BlobBackend:
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise ConfigError(
            f"{url} needs the Azure extra: pip install 'datasets_client[azure]'") from exc
    account, container, prefix = parse(url)
    connection = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection:
        service = BlobServiceClient.from_connection_string(connection)
    else:
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise ConfigError(
                f"{url} needs the Azure extra: pip install 'datasets_client[azure]'"
            ) from exc
        service = BlobServiceClient(f"https://{account}.blob.core.windows.net",
                                    credential=DefaultAzureCredential())
    return BlobBackend(AzureStore(service.get_container_client(container), prefix), url)
