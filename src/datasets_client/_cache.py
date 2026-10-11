"""The local cache: `<root>/<backend>/<name>/v<N>/<path>`, each file verified.

**A file is in the cache only once its bytes matched the listing.** It is
written to a temporary file in the same directory, hashed while it is written,
and moved into place with `os.replace` only if the size and sha256 agree - so a
reader never sees a half-written or wrong file, and two processes filling the
same cache at once each produce a whole file.

**Every destination is resolved and checked against the root before a byte is
written.** The listing came from a backend, and a backend's listing can be
written by any visitor of a UI that uploads; `_layout.check_path` refuses the
obvious escapes and this is the second lock, for the ones a grammar cannot see
(a symbolic link already sitting in the cache, a case-folding filesystem).

Beside each version directory sits `v<N>.listing.json`, the listing that
version was filled from. A cached file counts only if that listing still gives
the same sha256 for it: a folder or a container that rewrote a version in place
reads as a miss, not as a hit on stale bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from datasets_client import _layout
from datasets_client._layout import FileEntry, Version
from datasets_client.errors import IntegrityError, UnsafePath

Sink = Callable[[bytes], None]


def default_root() -> Path:
    """`DATASETS_CACHE`, else the temp directory. A deployment with a persistent
    volume points `DATASETS_CACHE` at it so a restart is not a cold read."""
    configured = os.environ.get("DATASETS_CACHE")
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "datasets-cache"


class Cache:
    def __init__(self, root: str | os.PathLike, backend_id: str) -> None:
        self.root = Path(root)
        # One subtree per backend URL: the same name in two backends is two
        # datasets, and must not answer each other's reads.
        self.base = self.root / hashlib.sha256(backend_id.encode()).hexdigest()[:16]

    # --- where things go -------------------------------------------------

    def _contained(self, path: Path) -> Path:
        root = self.root.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise UnsafePath(f"{path} resolves outside the cache root {root}")
        return resolved

    def dataset_dir(self, name: str) -> Path:
        _layout.check_name(name)
        return self._contained(self.base.joinpath(*name.split("/")))

    def version_dir(self, name: str, version: int) -> Path:
        return self._contained(self.dataset_dir(name) / f"v{version}")

    def _manifest(self, name: str, version: int) -> Path:
        return self.dataset_dir(name) / f"v{version}.{_layout.LISTING_FILE}"

    def file_path(self, name: str, version: int, path: str) -> Path:
        _layout.check_path(path)
        return self._contained(self.version_dir(name, version).joinpath(*path.split("/")))

    # --- reading ---------------------------------------------------------

    def _known(self, name: str, version: int) -> dict[str, str]:
        try:
            doc = json.loads(self._manifest(name, version).read_bytes())
            return {f["path"]: f["sha256"] for f in doc["files"]}
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def have(self, v: Version, entry: FileEntry) -> Path | None:
        """The cached file, if it is this version's and whole; else None."""
        target = self.file_path(v.name, v.version, entry.path)
        if self._known(v.name, v.version).get(entry.path) != entry.sha256:
            return None
        try:
            if target.is_file() and target.stat().st_size == entry.size_bytes:
                return target
        except OSError:
            return None
        return None

    def newest(self, name: str, path: str | None) -> tuple[int, Path] | None:
        """The newest cached version holding `path` (or every file, for a tree)."""
        try:
            entries = os.listdir(self.dataset_dir(name))
        except OSError:
            return None
        versions = sorted((v for v in map(_layout.version_dir, entries) if v), reverse=True)
        for version in versions:
            known = self._known(name, version)
            if not known:
                continue
            wanted = [path] if path is not None else list(known)
            if path is not None and path not in known:
                continue
            if all(self.file_path(name, version, p).is_file() for p in wanted):
                vdir = self.version_dir(name, version)
                return version, (self.file_path(name, version, path) if path else vdir)
        return None

    # --- writing ---------------------------------------------------------

    def begin(self, v: Version) -> None:
        """Record which listing this version directory is being filled from.
        A different listing for the same version empties the directory first."""
        manifest = self._manifest(v.name, v.version)
        body = _layout.listing_bytes(v.name, v.version, v.files)
        try:
            if manifest.read_bytes() == body:
                return
        except OSError:
            pass
        shutil.rmtree(self.version_dir(v.name, v.version), ignore_errors=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(manifest, body)

    def store(self, v: Version, entry: FileEntry, fill: Callable[[Sink], None]) -> Path:
        """Fill one file through `fill(sink)`, verify it, and move it into place."""
        target = self.file_path(v.name, v.version, entry.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._contained(target.parent)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".part-")
        h, size = hashlib.sha256(), 0
        try:
            with os.fdopen(fd, "wb") as out:
                def sink(chunk: bytes) -> None:
                    nonlocal size
                    h.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
                fill(sink)
            if size != entry.size_bytes or h.hexdigest() != entry.sha256:
                raise IntegrityError(
                    f"{v.name}@{v.version}/{entry.path}: got {size} bytes with sha256 "
                    f"{h.hexdigest()[:12]}..., the listing says {entry.size_bytes} bytes "
                    f"and {entry.sha256[:12]}...; discarded", status=None)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return target

    # --- forgetting ------------------------------------------------------

    def forget(self, name: str, versions) -> None:
        """Delete these versions of `name` from the cache (retracted ones)."""
        for version in versions:
            shutil.rmtree(self.version_dir(name, version), ignore_errors=True)
            try:
                self._manifest(name, version).unlink()
            except OSError:
                pass

    def evict_except(self, name: str, keep: int) -> None:
        """A new version means new keys; every other version of `name` goes.
        Errors are ignored - a file another process holds open on Windows is
        evicted next time."""
        try:
            entries = os.listdir(self.dataset_dir(name))
        except OSError:
            return
        stale = set()
        for entry in entries:
            version = _layout.version_dir(entry.split(".", 1)[0])
            if version is not None and version != keep:
                stale.add(version)
        self.forget(name, sorted(stale))


def _atomic_write(target: Path, body: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".part-")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(body)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
