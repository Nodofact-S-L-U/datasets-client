"""The layout over any key/value store: a folder, an Azure container, an S3 prefix.

Three backends hold the same tree (`_layout`), so they share one implementation
of it and differ only in five primitives - `BlobStore` below. Everything with a
rule in it (which version is latest, when a version exists, what a listing must
say about itself) is here once.

What these backends do NOT have, stated because the gateway backend does: grants,
an issuance log, pointers for moved names, retraction. A folder or a container
is readable by whoever can read it, and a version is removed by deleting it.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from datasets_client import _layout
from datasets_client._layout import DatasetInfo, FileEntry, Version
from datasets_client.errors import DatasetUnavailable, Refused, UnsafePath

Sink = Callable[[bytes], None]
_LISTING_KEY = re.compile(r"^(?P<name>.+)/v(?P<version>[1-9][0-9]{0,8})/listing\.json$")


class AlreadyExists(Exception):
    """`put(..., create_only=True)` found the key already there."""


class BlobStore(Protocol):
    """Five primitives over keys relative to the backend's root. Each raises
    `BackendUnreachable` when it cannot ASK, and reports absence as absence."""

    def children(self, prefix: str) -> list[str]:
        """The names one level below `prefix` (which ends in `/`)."""

    def keys(self, prefix: str) -> Iterator[str]:
        """Every key under `prefix`, recursively."""

    def read(self, key: str) -> bytes | None:
        """A small object whole, or None when there is none."""

    def download(self, key: str, sink: Sink) -> None:
        """Stream an object into `sink`; `DatasetUnavailable` when absent."""

    def put(self, key: str, source: Path | bytes, *, create_only: bool = False) -> None:
        """Write an object. `create_only` raises `AlreadyExists` instead of
        replacing - the listing is written that way, so two writers of the
        same version cannot both believe they published it."""


class BlobBackend:
    """The dataset operations, on any `BlobStore`."""

    writes_listing = True

    def __init__(self, store: BlobStore, url: str) -> None:
        self.store = store
        self.url = url

    # --- reading -------------------------------------------------------------

    def _versions(self, name: str) -> list[int]:
        found = (_layout.version_dir(c) for c in self.store.children(f"{name}/"))
        return sorted((v for v in found if v), reverse=True)

    def _read_version(self, name: str, version: int) -> Version | None:
        raw = self.store.read(f"{name}/v{version}/{_layout.LISTING_FILE}")
        if raw is None:
            return None
        return _layout.parse_listing(raw, name=name, version=version)

    def resolve(self, name: str, version: int | None = None) -> Version:
        _layout.check_name(name)
        candidates = [version] if version is not None else self._versions(name)
        for v in candidates:
            got = self._read_version(name, v)
            if got is not None:
                return Version(name=got.name, version=got.version, files=got.files,
                               metadata=got.metadata, requested=name)
        where = f"{name}@{version}" if version is not None else name
        raise DatasetUnavailable(f"{where} is not in {self.url}", status=404)

    def latest(self, name: str) -> int | None:
        try:
            return self.resolve(name).version
        except DatasetUnavailable:
            return None

    def list(self, prefix: str = "") -> list[DatasetInfo]:
        """Every dataset whose name starts with `prefix`, newest version each.

        Found by its `listing.json` keys, then each candidate's listing is READ
        and must name itself: `ns/a/v1/v2/listing.json` is version 2 of
        `ns/a/v1` or a file inside `ns/a`'s version 1, and only the document
        can say which.
        """
        base = prefix.rsplit("/", 1)[0] + "/" if "/" in prefix else ""
        best: dict[str, list[int]] = {}
        for key in self.store.keys(base):
            m = _LISTING_KEY.match(key)
            if not m or not m.group("name").startswith(prefix):
                continue
            best.setdefault(m.group("name"), []).append(int(m.group("version")))
        out = []
        for name in sorted(best):
            try:
                _layout.check_name(name)
            except Exception:  # noqa: BLE001,S112 - a key that is not a dataset name
                continue
            for version in sorted(best[name], reverse=True):
                try:
                    got = self._read_version(name, version)
                except UnsafePath:
                    continue
                if got is not None:
                    out.append(DatasetInfo(
                        name=name, latest_version=version, file_count=len(got.files),
                        total_bytes=got.total_bytes, metadata=got.metadata))
                    break
        return out

    def download(self, v: Version, entry: FileEntry, sink: Sink) -> None:
        self.store.download(f"{v.name}/v{v.version}/{entry.path}", sink)

    # --- writing -------------------------------------------------------------

    def write_version(self, name: str, version: int, files: tuple[FileEntry, ...],
                      source: Callable[[FileEntry], Path], metadata: Mapping) -> int:
        """Write one version: every file, then the listing, create-only.

        The listing last is what makes a version appear whole or not at all; a
        crash part-way leaves files nobody can see, and the next writer of that
        version overwrites them.
        """
        for entry in files:
            self.store.put(f"{name}/v{version}/{entry.path}", source(entry))
        try:
            self.store.put(f"{name}/v{version}/{_layout.LISTING_FILE}",
                           _layout.listing_bytes(name, version, files, metadata),
                           create_only=True)
        except AlreadyExists:
            raise Refused(f"{name}@{version} already exists in {self.url}; another "
                          "writer published it first", status=409) from None
        return version

    def create(self, local_dir: Path, files: tuple[FileEntry, ...], *,
               name: str | None, namespace: str | None, parent: int | None,
               metadata: Mapping | None, label: str | None, breaking: bool = False,
               user_token: str | None = None) -> dict:
        """A new dataset, or the next version of one, from a local directory.

        With `namespace` and no `name`, the name is assigned the way the gateway
        assigns one, under `<ns>/uploads/local/`. A new version must name the
        version it builds on, as on a gateway: a stale `parent` is `409`.
        """
        if name is None:
            ns = _layout.check_namespace(namespace)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")  # noqa: UP017
            name = f"{ns}/{_layout.UPLOADS}/local/{stamp}-{secrets.token_hex(4)}"
        _layout.check_name(name)
        current = self.latest(name)
        if current != parent:
            raise Refused(
                f"{name}: the latest version is {current}, not {parent}. Resolve "
                "again and name the version this builds on.", status=409)
        if current is None:
            meta = _layout.check_metadata(metadata)
            self._check_leaf(name)
        else:
            before = self.resolve(name).metadata
            meta = dict(metadata) if metadata else before
            had, has = _layout.data_class_of(before), _layout.data_class_of(meta)
            if had is not None and has is not None and has != had:
                # The class says what kind of data the name holds; a version
                # that changes it is not a correction, so it is not silent.
                if not breaking:
                    raise Refused(
                        f"{name} holds {had} data and this version says {has}. A "
                        "change of class is a breaking change: pass breaking if "
                        "it is meant.", status=409)
                meta = _layout.check_metadata(meta)
        if label:
            meta = meta | {"label": label}
        version = self.write_version(
            name, (current or 0) + 1, files,
            lambda e: local_dir.joinpath(*e.path.split("/")), meta)
        return {"name": name, "version": version}

    def _check_leaf(self, name: str) -> None:
        ns = name.split("/", 1)[0]
        for info in self.list(ns + "/"):
            other = info.name
            if other.startswith(name + "/") or name.startswith(other + "/"):
                raise Refused(f"{name} would nest with {other}; a dataset is a leaf",
                              status=409)

    def move(self, *args, **kwargs):
        raise Refused(f"move is an operator action on a dataset gateway; {self.url} has no index "
                      "to keep a pointer in. Copy the version and delete the old one.",
                      status=400)

    def retract(self, *args, **kwargs):
        raise Refused(f"retract is an operator action on a dataset gateway; in {self.url}, delete "
                      "the version's directory.", status=400)
