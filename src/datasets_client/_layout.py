"""Names, paths and `listing.json` - the layout every backend shares.

```
<name>/v<N>/<path>
<name>/v<N>/listing.json
```

`listing.json` is `{"name", "version", "files": [{"path", "size_bytes",
"sha256"}]}` and, outside a dataset gateway (which keeps it in its index),
`metadata`. A version EXISTS when its `listing.json` does: a writer puts every
file first and the listing last, so a half-written version is invisible.

**The grammar here mirrors the gateway's, deliberately, and does not import
it.** This package ships into products that never see the gateway's code, and
the listing it reads may come from a blob container anybody with the account
key can write. So every name and path is checked HERE, on the machine that is
about to write files, whatever the backend already promised.
The gateway's own test suite runs both grammars over one corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from datasets_client.errors import DatasetError, DatasetUnavailable, Refused, UnsafePath

LISTING_FILE = "listing.json"
UPLOADS = "uploads"
MAX_NAME = 200
MAX_SEGMENT = 64
MAX_DEPTH = 8
MAX_PATH = 512
MAX_PATH_SEGMENT = 255
MAX_FILES = 10_000
REQUIRED_METADATA = ("description", "source", "ci_allowed", "synthetic")
CHUNK = 1024 * 1024

_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_DIR = re.compile(r"^v([1-9][0-9]{0,8})$")
_WINDOWS_DEVICES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


# --- names ------------------------------------------------------------------------


def check_name(name: object) -> str:
    """`<namespace>/<part>[/<part>...]`; each part `[a-z0-9][a-z0-9._-]*`.

    Raises `DatasetError` (not `UnsafePath`): a bad name is a caller's mistake,
    caught before any backend is asked.
    """
    if not isinstance(name, str) or not name or len(name) > MAX_NAME:
        raise DatasetError(f"{name!r} is not a dataset name", status=400)
    parts = name.split("/")
    if not 2 <= len(parts) <= MAX_DEPTH:
        raise DatasetError(
            f"{name!r}: a dataset name is `<namespace>/<name>`, at most "
            f"{MAX_DEPTH} parts", status=400)
    for seg in parts:
        if len(seg) > MAX_SEGMENT or not _SEGMENT.match(seg):
            raise DatasetError(
                f"{name!r}: each part is lower-case letters, digits, '.', '_' and "
                "'-', starting with a letter or a digit", status=400)
    if parts[0] == UPLOADS or any(s == UPLOADS for s in parts[2:]):
        raise DatasetError(f"{name!r}: '{UPLOADS}' may only be the second part",
                           status=400)
    if parts[1] == UPLOADS and len(parts) < 4:
        raise DatasetError(f"{name!r}: an upload is `<ns>/uploads/<writer>/<stamp>`",
                           status=400)
    return name


def check_namespace(ns: object) -> str:
    if (not isinstance(ns, str) or len(ns) > MAX_SEGMENT or not _SEGMENT.match(ns)
            or ns == UPLOADS):
        raise DatasetError(f"{ns!r} is not a namespace", status=400)
    return ns


def parse_ref(ref: object) -> tuple[str, int | None]:
    """`name` or `name@<version>`."""
    if not isinstance(ref, str):
        raise DatasetError("a dataset reference is a string", status=400)
    name, sep, version = ref.partition("@")
    if not sep:
        return check_name(name), None
    if not version.isdigit() or version.startswith("0") or len(version) > 9:
        raise DatasetError(f"{ref!r}: a version is a whole number from 1, as `name@3`",
                           status=400)
    return check_name(name), int(version)


def version_dir(entry: str) -> int | None:
    """`v12` -> 12; anything else -> None."""
    m = _VERSION_DIR.match(entry)
    return int(m.group(1)) if m else None


# --- paths ------------------------------------------------------------------------


def check_path(path: object) -> str:
    """One path inside a version. Refuses anything that could escape a directory
    or alias a file on Windows or macOS - with `UnsafePath`, because a path that
    fails here arrived in a LISTING, which is data somebody else wrote."""
    if not isinstance(path, str) or not path or len(path) > MAX_PATH:
        raise UnsafePath(f"{path!r} is not a usable file path")
    if path.startswith("/") or "\\" in path:
        raise UnsafePath(f"{path!r} is absolute or uses a backslash")
    for seg in path.split("/"):
        if seg in ("", ".", ".."):
            raise UnsafePath(f"{path!r} has an empty, '.' or '..' part")
        if len(seg) > MAX_PATH_SEGMENT or not _PATH_SEGMENT.match(seg):
            raise UnsafePath(f"{path!r} is not a usable file path")
        if seg.endswith("."):
            raise UnsafePath(f"{path!r}: a part may not end with '.'")
        if seg.split(".", 1)[0].lower() in _WINDOWS_DEVICES:
            raise UnsafePath(f"{path!r}: {seg!r} is a device name on Windows")
    if path == LISTING_FILE:
        raise UnsafePath(f"'{LISTING_FILE}' at the top of a dataset is the listing itself")
    return path


@dataclass(frozen=True)
class FileEntry:
    path: str
    size_bytes: int
    sha256: str
    url: str | None = None
    """A presigned GET, from the gateway backend only."""


def check_files(files: object) -> tuple[FileEntry, ...]:
    """A whole listing, checked as the gateway checks one, sorted by path."""
    if not isinstance(files, list) or not files or len(files) > MAX_FILES:
        raise UnsafePath(f"a listing is 1 to {MAX_FILES} files")
    out: list[FileEntry] = []
    folded: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, Mapping):
            raise UnsafePath("each listed file is {path, size_bytes, sha256}")
        path = check_path(entry.get("path"))
        size, digest = entry.get("size_bytes"), entry.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise UnsafePath(f"{path!r}: size_bytes is not a whole number")
        if not isinstance(digest, str) or not _SHA256.match(digest):
            raise UnsafePath(f"{path!r}: sha256 is not 64 lower-case hex characters")
        if path.casefold() in folded:
            raise UnsafePath(f"{path!r} and {folded[path.casefold()]!r} are one file "
                             "on a case-insensitive disk")
        folded[path.casefold()] = path
        url = entry.get("url")
        out.append(FileEntry(path, size, digest, url if isinstance(url, str) else None))
    paths = sorted(f.path for f in out)
    for a, b in zip(paths, paths[1:], strict=False):
        if b.startswith(a + "/"):
            raise UnsafePath(f"{a!r} is a file and also the directory of {b!r}")
    return tuple(sorted(out, key=lambda f: f.path))


def check_metadata(meta: object) -> dict:
    """The four keys a dataset must carry when it is created; `synthetic` must be
    exactly `true`. The gateway checks this too; checking it here as well means
    a copy into a folder or a blob container cannot drop the attestation."""
    if not isinstance(meta, Mapping):
        raise Refused("metadata must be an object", status=400)
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise Refused(f"metadata is missing {', '.join(missing)}", status=400)
    if meta["synthetic"] is not True:
        raise Refused("metadata.synthetic must be true: only synthetic data is held",
                      status=422)
    if not isinstance(meta["ci_allowed"], bool):
        raise Refused("metadata.ci_allowed must be true or false", status=400)
    for key in ("description", "source"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise Refused(f"metadata.{key} must be a non-empty string", status=400)
    return dict(meta)


# --- versions ---------------------------------------------------------------------


@dataclass(frozen=True)
class Version:
    """One resolved version: what a read is served from."""

    name: str
    version: int
    files: tuple[FileEntry, ...]
    metadata: dict = field(default_factory=dict)
    requested: str | None = None
    pointers: tuple[str, ...] = ()
    retracted: tuple[int, ...] = ()
    published_at: str | None = None

    def file(self, path: str) -> FileEntry:
        for f in self.files:
            if f.path == path:
                return f
        raise DatasetUnavailable(f"{self.name}@{self.version} has no file {path!r}",
                                 status=404)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


@dataclass(frozen=True)
class DatasetInfo:
    """One row of a listing: enough for a picker."""

    name: str
    latest_version: int | None
    file_count: int | None
    total_bytes: int | None
    metadata: dict
    moved_to: str | None = None
    published_at: str | None = None


def listing_bytes(name: str, version: int, files: Iterable[FileEntry],
                  metadata: Mapping | None = None) -> bytes:
    """`listing.json`, byte-stable. Without metadata it is byte-for-byte what
    the gateway writes, so a listing copied out of a gateway and back compares equal.
    """
    doc: dict = {"name": name, "version": version,
                 "files": [{"path": f.path, "size_bytes": f.size_bytes, "sha256": f.sha256}
                           for f in sorted(files, key=lambda f: f.path)]}
    if metadata:
        doc["metadata"] = dict(metadata)
    return (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode("utf-8")


def parse_listing(raw: bytes, *, name: str, version: int) -> Version | None:
    """A `listing.json` read from a backend, or None when it is not THIS
    version's listing.

    It must parse and name this dataset and version: a file that happens to be
    called `listing.json` deeper inside another dataset fails that, and is not
    a version - not an error either. A listing that DOES name itself and lists
    an unsafe path raises `UnsafePath`.
    """
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("name") != name or doc.get("version") != version:
        return None
    meta = doc.get("metadata")
    return Version(name=name, version=version, files=check_files(doc.get("files")),
                   metadata=meta if isinstance(meta, dict) else {})


# --- a local directory, as the source of an upload -------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def scan(local_dir: str | os.PathLike) -> tuple[FileEntry, ...]:
    """Every regular file under `local_dir`, hashed, as a listing.

    Symbolic links are refused rather than followed: an upload is what is IN the
    directory, and a link to `~/.ssh` is not. So is a top-level `listing.json`,
    which would be read as the listing itself.
    """
    root = Path(local_dir)
    if not root.is_dir():
        raise DatasetError(f"{root} is not a directory", status=400)
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for d in list(dirnames):
            if (Path(dirpath) / d).is_symlink():
                raise UnsafePath(f"{Path(dirpath) / d} is a symbolic link; not uploaded")
        for fn in filenames:
            full = Path(dirpath) / fn
            if full.is_symlink() or not full.is_file():
                raise UnsafePath(f"{full} is not a regular file; not uploaded")
            rel = full.relative_to(root).as_posix()
            files.append({"path": rel, "size_bytes": full.stat().st_size,
                          "sha256": sha256_file(full)})
    if not files:
        raise DatasetError(f"{root} holds no files", status=400)
    return check_files(files)
