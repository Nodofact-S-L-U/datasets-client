"""Plain S3: `s3://<bucket>/<prefix>`, with the ambient AWS identity.

For a deployment on AWS that reads the bucket directly. Behind a dataset
gateway an app usually holds no S3 grant at all - the gateway signs URLs
instead - and uses the gateway backend. `boto3` is the `[s3]` extra, imported on first use.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from datasets_client.backends._blob import AlreadyExists, BlobBackend, Sink
from datasets_client.errors import BackendUnreachable, ConfigError, DatasetUnavailable

_MISSING = {"NoSuchKey", "404", "NotFound"}


def parse(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if not parsed.netloc:
        raise ConfigError(f"{url}: expected s3://<bucket>[/<prefix>]")
    prefix = parsed.path.strip("/")
    return parsed.netloc, (prefix + "/") if prefix else ""


class S3Store:
    """`BlobStore` on one bucket, under one prefix."""

    def __init__(self, client, bucket: str, prefix: str) -> None:
        self.s3 = client
        self.bucket = bucket
        self.prefix = prefix
        from botocore.exceptions import BotoCoreError, ClientError
        self._errors = (BotoCoreError, ClientError)
        self._client_error = ClientError

    def _code(self, exc: Exception) -> str:
        return getattr(exc, "response", {}).get("Error", {}).get("Code", "")

    def _unreachable(self, what: str, exc: Exception) -> BackendUnreachable:
        return BackendUnreachable(f"S3: could not {what}: {exc}")

    def children(self, prefix: str) -> list[str]:
        full = self.prefix + prefix
        try:
            out = []
            pages = self.s3.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=full, Delimiter="/")
            for page in pages:
                for cp in page.get("CommonPrefixes", []):
                    out.append(cp["Prefix"][len(full):].rstrip("/"))
            return out
        except self._errors as exc:
            raise self._unreachable(f"list {full}", exc) from exc

    def keys(self, prefix: str) -> Iterator[str]:
        full = self.prefix + prefix
        try:
            pages = self.s3.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket, Prefix=full)
            for page in pages:
                for obj in page.get("Contents", []):
                    yield obj["Key"][len(self.prefix):]
        except self._errors as exc:
            raise self._unreachable(f"list {full}", exc) from exc

    def read(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()
        except self._errors as exc:
            if self._code(exc) in _MISSING:
                return None
            raise self._unreachable(f"read {key}", exc) from exc

    def download(self, key: str, sink: Sink) -> None:
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"]
            for chunk in body.iter_chunks(1024 * 1024):
                sink(chunk)
        except self._errors as exc:
            if self._code(exc) in _MISSING:
                raise DatasetUnavailable(f"{key} is listed but not in the bucket",
                                         status=404) from None
            raise self._unreachable(f"read {key}", exc) from exc

    def put(self, key: str, source: Path | bytes, *, create_only: bool = False) -> None:
        extra = {"IfNoneMatch": "*"} if create_only else {}
        try:
            if isinstance(source, bytes):
                self.s3.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=source,
                                   **extra)
            else:
                with open(source, "rb") as body:
                    self.s3.put_object(Bucket=self.bucket, Key=self.prefix + key,
                                       Body=body, **extra)
        except self._errors as exc:
            if self._code(exc) in ("PreconditionFailed", "ConditionalRequestConflict"):
                raise AlreadyExists(key) from None
            raise self._unreachable(f"write {key}", exc) from exc


def backend(url: str) -> BlobBackend:
    try:
        import boto3
    except ImportError as exc:
        raise ConfigError(f"{url} needs the S3 extra: pip install 'datasets_client[s3]'"
                          ) from exc
    bucket, prefix = parse(url)
    return BlobBackend(S3Store(boto3.client("s3"), bucket, prefix), url)
