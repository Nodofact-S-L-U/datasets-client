"""`Datasets`: the one object product code holds.

```python
from datasets_client import Datasets
ds = Datasets()                                   # reads DATASETS_URL
path = ds.file("example/dataset01", "data.csv")
root = ds.tree("example/dataset01")
for d in ds.list("example/"): ...
ds.upload(local_dir, namespace="example", label="...", metadata={...})
```

**Every read resolves the latest version at that moment, then serves it from
the cache.** "Always the latest" is exact - a new version is used from the next
call on - and a repeat read of an unchanged version costs one small request.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from datasets_client import _layout
from datasets_client._cache import Cache, default_root
from datasets_client._layout import DatasetInfo, FileEntry, Version
from datasets_client.backends import open_backend
from datasets_client.errors import BackendUnreachable, DatasetError, IntegrityError, Refused

URL_VARIABLE = "DATASETS_URL"


@dataclass(frozen=True)
class Fetched:
    """What a read served, and from which version.

    `stale` is True only when the caller passed `allow_stale=True` and the
    backend could not be reached: `path` is then an OLDER cached version, named
    by `version`, and `error` says why the latest could not be had.
    """

    path: Path
    name: str
    version: int
    requested: str
    stale: bool = False
    error: DatasetError | None = None


class Datasets:
    """Read and write datasets in whatever backend `DATASETS_URL` names.

    Nothing is contacted until the first read, so constructing one with no URL
    configured is fine; the read raises `ConfigError`.
    """

    def __init__(self, url: str | None = None, *, token: str | None = None,
                 cache_dir: str | os.PathLike | None = None, transport=None) -> None:
        self.url = url if url is not None else os.environ.get(URL_VARIABLE)
        self._token = token
        self._transport = transport
        self._cache_dir = cache_dir
        self._backend = None
        self._cache: Cache | None = None

    # --- plumbing -----------------------------------------------------------

    @property
    def backend(self):
        if self._backend is None:
            self._backend = open_backend(self.url, token=self._token,
                                         transport=self._transport)
        return self._backend

    @property
    def cache(self) -> Cache:
        if self._cache is None:
            root = Path(self._cache_dir) if self._cache_dir else default_root()
            self._cache = Cache(root, self.backend.url)
        return self._cache

    # --- reading ------------------------------------------------------------

    def list(self, prefix: str = "") -> list[DatasetInfo]:
        """Every dataset under `prefix` this caller may read. Never `[]` for
        "could not ask": that raises."""
        return self.backend.list(prefix)

    def resolve(self, name: str, *, version: int | None = None) -> Version:
        """The latest version's listing (or a pinned one, where the backend
        allows pinning - CI and operators on a gateway, any caller elsewhere)."""
        return self.backend.resolve(name, version)

    def file(self, name: str, path: str) -> Path:
        """One file of the latest version, as a local path."""
        return self.fetch(name, path).path

    def tree(self, name: str) -> Path:
        """Every file of the latest version, as a local directory."""
        return self.fetch(name).path

    def fetch(self, name: str, path: str | None = None, *, version: int | None = None,
              allow_stale: bool = False) -> Fetched:
        """`file` and `tree`, saying which version they served.

        `allow_stale=True` falls back to the newest cached version when - and
        only when - the backend cannot be reached. Not granted, retracted or a
        hash mismatch still raise: serving an old copy of data somebody revoked
        or withdrew is not a fallback, it is the failure.
        """
        _layout.check_name(name)
        if path is not None:
            _layout.check_path(path)
        try:
            v = self.resolve(name, version=version)
        except BackendUnreachable as exc:
            if not allow_stale:
                raise
            cached = self.cache.newest(name, path)
            if cached is None:
                raise
            return Fetched(path=cached[1], name=name, version=cached[0], requested=name,
                           stale=True, error=exc)
        self.cache.forget(v.name, v.retracted)
        v = self._fill(v, path, name=name, version=version)
        self.cache.evict_except(v.name, keep=v.version)
        target = (self.cache.file_path(v.name, v.version, path) if path is not None
                  else self.cache.version_dir(v.name, v.version))
        return Fetched(path=target, name=v.name, version=v.version, requested=name)

    def _fill(self, v: Version, path: str | None, *, name: str,
              version: int | None) -> Version:
        """Download what the cache lacks; return the version actually served.

        A URL that died (`403`: the signing credentials rotated) is answered by
        resolving again, ONCE. If a newer version appeared meanwhile, that is
        the one served - "always the latest" holds for the retry too.
        """
        from datasets_client.backends.http_gateway import UrlExpired

        for attempt in range(2):
            wanted = [v.file(path)] if path is not None else list(v.files)
            self.cache.begin(v)
            try:
                for entry in wanted:
                    if self.cache.have(v, entry) is None:
                        self.cache.store(v, entry, lambda sink, e=entry, v=v:
                                         self.backend.download(v, e, sink))
                return v
            except UrlExpired:
                if attempt:
                    raise BackendUnreachable(
                        f"{v.name}@{v.version}: download URLs kept dying after a "
                        "fresh resolve") from None
                v = self.resolve(name, version=version)
                self.cache.forget(v.name, v.retracted)
        return v

    # --- writing ------------------------------------------------------------

    def upload(self, local_dir: str | os.PathLike, *, namespace: str,
               metadata: Mapping, label: str | None = None,
               user_token: str | None = None) -> dict:
        """A NEW dataset under `<namespace>/uploads/`, named by the backend.

        `user_token` is the identity of the person behind the upload when there
        is one (behind a gateway, the identity it handed the app); leave it out
        for a service with no login. Returns `{"name", "version"}`.
        """
        root = Path(local_dir)
        return self.backend.create(root, _layout.scan(root), name=None,
                                   namespace=_layout.check_namespace(namespace),
                                   parent=None, metadata=metadata, label=label,
                                   user_token=user_token)

    def push(self, local_dir: str | os.PathLike, name: str, *, parent: int | None = None,
             metadata: Mapping | None = None, label: str | None = None,
             breaking: bool = False) -> dict:
        """A dataset under a curated name, or its next version (operator)."""
        root = Path(local_dir)
        return self.backend.create(root, _layout.scan(root), name=_layout.check_name(name),
                                   namespace=None, parent=parent, metadata=metadata,
                                   label=label, breaking=breaking)

    def pull(self, ref: str, dest: str | os.PathLike) -> Version:
        """Copy one version into a local folder, in the layout - what a
        standalone run's `DATASETS_URL` then points at. Retracted versions the
        source names are deleted from the folder."""
        name, version = _layout.parse_ref(ref)
        Path(dest).mkdir(parents=True, exist_ok=True)
        return copy(self, Datasets(str(dest), cache_dir=self._cache_dir), name, version)

    def move(self, source: str, dest: str, *, breaking: bool = False,
             label: str | None = None) -> dict:
        """Operator, gateway only: a version to a curated name or into an existing
        dataset as its next version; the source keeps a forward pointer."""
        return self.backend.move(source, dest, breaking=breaking, label=label)

    def retract(self, name: str, version: int, reason: str) -> dict:
        """Operator, gateway only: tombstone one version and delete its files."""
        return self.backend.retract(name, version, reason)


def copy(src: Datasets, dst: Datasets, name: str, version: int | None = None, *,
         breaking: bool = False, label: str | None = None) -> Version:
    """One version from any backend to any other.

    Into a folder, an Azure container or an S3 prefix the version keeps its
    NUMBER, so the copy is a mirror and "latest" means the same thing on both
    sides. Into a gateway, the gateway numbers it: it becomes the next version of
    `name` there, with `breaking` required if the layout changed.
    """
    v = src.resolve(name, version=version)
    with tempfile.TemporaryDirectory(prefix="datasets-copy-") as tmp:
        staging = Path(tmp)
        v = _download_all(src, v, staging)
        if dst.backend.writes_listing:
            if dst.backend.latest(v.name) is not None:
                try:
                    have = dst.backend.resolve(v.name, v.version)
                except DatasetError:
                    have = None
                if have is not None:
                    if [(f.path, f.sha256) for f in have.files] != [
                            (f.path, f.sha256) for f in v.files]:
                        raise IntegrityError(
                            f"{v.name}@{v.version} already exists in {dst.url} with "
                            "different contents; versions are immutable")
                    _forget_retracted(dst, v)
                    return v
            dst.backend.write_version(v.name, v.version, v.files,
                                      lambda e: staging.joinpath(*e.path.split("/")),
                                      v.metadata)
            _forget_retracted(dst, v)
        else:
            dst.backend.create(staging, v.files, name=v.name, namespace=None,
                               parent=dst.backend.latest(v.name), metadata=v.metadata,
                               label=label, breaking=breaking)
    return v


def _forget_retracted(dst: Datasets, v: Version) -> None:
    """In a folder, a retracted version's directory is removed; elsewhere the
    copy says nothing and an operator deletes it."""
    root = getattr(dst.backend.store, "root", None)
    if root is None:
        return
    for gone in v.retracted:
        shutil.rmtree(Path(root).joinpath(*v.name.split("/"), f"v{gone}"),
                      ignore_errors=True)


def _download_all(src: Datasets, v: Version, staging: Path) -> Version:
    """Every file of `v` into `staging`, verified. A URL that died (a copy of
    many files can outlast 900 seconds) is re-resolved once, pinned to the same
    version where the caller may pin, and must come back as that version."""
    from datasets_client.backends.http_gateway import UrlExpired

    done: set[str] = set()
    for attempt in range(2):
        try:
            for entry in v.files:
                if entry.path in done:
                    continue
                target = staging.joinpath(*entry.path.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                _download_verified(src, v, entry, target)
                done.add(entry.path)
            return v
        except UrlExpired:
            if attempt:
                raise BackendUnreachable(f"{v.name}@{v.version}: download URLs kept "
                                         "dying after a fresh resolve") from None
            try:
                fresh = src.resolve(v.name, version=v.version)
            except Refused:
                fresh = src.resolve(v.name)
            if fresh.version != v.version:
                raise BackendUnreachable(
                    f"{v.name} moved from version {v.version} to {fresh.version} during "
                    "the copy; run it again") from None
            v = fresh
    return v


def _download_verified(src: Datasets, v: Version, entry: FileEntry, target: Path) -> None:
    h, size = hashlib.sha256(), 0
    with open(target, "wb") as out:
        def sink(chunk: bytes) -> None:
            nonlocal size
            h.update(chunk)
            size += len(chunk)
            out.write(chunk)
        src.backend.download(v, entry, sink)
    if size != entry.size_bytes or h.hexdigest() != entry.sha256:
        target.unlink()
        raise IntegrityError(f"{v.name}@{v.version}/{entry.path}: the bytes do not match "
                             "the listing; nothing was copied")
