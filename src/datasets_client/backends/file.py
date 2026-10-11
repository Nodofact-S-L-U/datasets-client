"""A local folder holding the layout: standalone runs, and CI after `pull`.

`DATASETS_URL=file:///data/datasets` or a bare path. `datasets_client pull`
fills one from any other backend, which is what lets an app run with no cloud
at all: the same code, the same names, the folder in place of the gateway.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlparse

from datasets_client._layout import CHUNK
from datasets_client.backends._blob import AlreadyExists, BlobBackend, Sink
from datasets_client.errors import BackendUnreachable, DatasetUnavailable, UnsafePath


def folder_of(url: str) -> Path:
    """`file:///C:/data` / `file:///data` / a bare path -> the folder."""
    if url.startswith("file:"):
        parsed = urlparse(url)
        if parsed.netloc not in ("", "localhost"):
            raise BackendUnreachable(f"{url}: a file URL names this machine only")
        path = unquote(parsed.path)
        # `file:///C:/data` parses to `/C:/data`; the leading slash is not part
        # of a Windows path.
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
        return Path(path)
    return Path(url)


class FolderStore:
    """`BlobStore` on a directory. Keys are `/`-separated and relative."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        target = self.root.joinpath(*key.split("/"))
        root = self.root.resolve()
        if not target.resolve().is_relative_to(root):
            raise UnsafePath(f"{key!r} resolves outside {root}")
        return target

    def _require_root(self) -> None:
        if not self.root.is_dir():
            raise BackendUnreachable(f"{self.root} is not a directory (DATASETS_URL)")

    def children(self, prefix: str) -> list[str]:
        self._require_root()
        try:
            return [e.name for e in os.scandir(self._path(prefix.rstrip("/")))
                    if e.is_dir()]
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise BackendUnreachable(f"cannot list {prefix} in {self.root}: {exc}") from exc

    def keys(self, prefix: str) -> Iterator[str]:
        self._require_root()
        start = self._path(prefix.rstrip("/")) if prefix else self.root

        def fail(exc: OSError) -> None:
            raise BackendUnreachable(f"cannot list {start}: {exc}") from exc

        if not start.is_dir():
            return
        for dirpath, _dirs, files in os.walk(start, onerror=fail):
            for fn in files:
                yield (Path(dirpath) / fn).relative_to(self.root).as_posix()

    def read(self, key: str) -> bytes | None:
        self._require_root()
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise BackendUnreachable(f"cannot read {key} in {self.root}: {exc}") from exc

    def download(self, key: str, sink: Sink) -> None:
        self._require_root()
        try:
            with open(self._path(key), "rb") as f:
                while chunk := f.read(CHUNK):
                    sink(chunk)
        except FileNotFoundError:
            raise DatasetUnavailable(f"{key} is listed but not in {self.root}",
                                     status=404) from None
        except OSError as exc:
            raise BackendUnreachable(f"cannot read {key} in {self.root}: {exc}") from exc

    def put(self, key: str, source: Path | bytes, *, create_only: bool = False) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".part-")
        try:
            with os.fdopen(fd, "wb") as out:
                if isinstance(source, bytes):
                    out.write(source)
                else:
                    with open(source, "rb") as src:
                        shutil.copyfileobj(src, out, CHUNK)
            if create_only:
                # A hard link fails if the name exists - the one atomic
                # "create if absent" every local filesystem offers.
                try:
                    os.link(tmp, target)
                except FileExistsError:
                    raise AlreadyExists(key) from None
                except OSError:
                    # A filesystem without hard links (FAT, some shares):
                    # check-then-replace, which is racy and says so here.
                    if target.exists():
                        raise AlreadyExists(key) from None
                    os.replace(tmp, target)
                    return
                os.unlink(tmp)
            else:
                os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def backend(url: str) -> BlobBackend:
    return BlobBackend(FolderStore(folder_of(url)), url)
