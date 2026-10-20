"""The dataset gateway: any `http(s)://` URL, the base of its datasets API.

The only module in this package that speaks HTTP to a gateway, and the only one
that reads a credential variable: `DATASETS_TOKEN`, the caller's bearer token.
Product code reads `DATASETS_URL` and nothing else.

**The gateway authorises; the bytes go straight to and from object storage.** A
resolve returns the listing and a presigned GET per file; an upload is `begin`
(the listing in, a presigned PUT per file out), the PUTs, then `commit`.
The gateway's published dataset protocol is the wire this speaks.

**Standard library only** (`urllib.request`), so the base install has no
dependencies at all - an image that only ever reads from a gateway or a folder
installs nothing else.

Two things a URL can do that a caller must handle, both handled here:

* **die early.** A presigned URL is signed with the gateway's role, and a role's
  credentials rotate, so a URL can answer `403` before its `expires_at`. A
  download that gets `403` re-resolves (`UrlExpired`, caught one level up); an
  upload asks for fresh URLs (`/-/urls/<id>`) and retries that file - or, for
  a file in parts, carries on from the part that failed. Every URL lives 900 s
  from `begin`, so a file in parts that takes longer also asks again between
  parts, before they lapse (#494).
* **be replayed.** A single-PUT URL carries `If-None-Match: *`, so a retry of a
  PUT that had in fact landed answers `412`. That is success: `commit` checks
  every object's size and checksum regardless.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

from datasets_client import _layout
from datasets_client._layout import CHUNK, DatasetInfo, FileEntry, Version
from datasets_client.errors import (
    BackendUnreachable,
    ConfigError,
    DatasetUnavailable,
    Refused,
    Retracted,
    UnsafePath,
)

Sink = Callable[[bytes], None]
TOKEN_VARIABLE = "DATASETS_TOKEN"  # noqa: S105 - a variable name, not a value
API_TIMEOUT = 60
TRANSFER_TIMEOUT = 300
COMMIT_TIMEOUT = 900
"""A commit copies every object server-side before it answers - seconds for the
few hundred MB, and the gateway's own budget is what bounds it."""
IDENTITY_HEADER = "X-Datasets-Identity"
"""The header a forwarded user identity travels in, to the dataset gateway."""
REFRESH_MARGIN = timedelta(seconds=30)
SEND_ATTEMPTS = 3
"""Refusals IN A ROW - with no part landing between them - before a file fails."""
_EXPIRING = object()
"""`_send` stopped between parts because the URLs are about to lapse: not a refusal."""
PAGE = 1000


class UrlExpired(Exception):
    """A presigned GET answered 403: resolve again, do not retry the URL."""


class Transport(Protocol):
    """HTTP, as this backend needs it. `UrllibTransport` below; a fake in tests."""

    def request(self, method: str, url: str, *, headers: Mapping[str, str],
                body=None, timeout: float) -> tuple[int, bytes]:
        """Send; return (status, body) for EVERY status. Raise
        `BackendUnreachable` only when nothing came back."""

    def stream(self, url: str, sink: Sink, *, timeout: float) -> int:
        """GET; feed a 200's body to `sink`, return the status."""


class UrllibTransport:
    def __init__(self) -> None:
        # No proxy handler: a deployment typically reaches its gateway and its
        # object storage over a private network, and an inherited HTTP_PROXY
        # would route both somewhere else entirely.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, url, *, headers, body=None, timeout):
        _http_only(url)
        req = urllib.request.Request(  # noqa: S310 - http(s) only, checked above
            url, data=body, method=method, headers=dict(headers))
        if body is not None and not req.has_header("Content-type"):
            req.add_header("Content-Type", "application/octet-stream")
        try:
            with self.opener.open(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendUnreachable(f"{method} {_redact(url)}: {exc}") from exc

    def stream(self, url, sink, *, timeout):
        _http_only(url)
        try:
            req = urllib.request.Request(url)  # noqa: S310 - http(s) only, checked above
            with self.opener.open(req, timeout=timeout) as resp:
                while chunk := resp.read(CHUNK):
                    sink(chunk)
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendUnreachable(f"GET {_redact(url)}: {exc}") from exc


def _http_only(url: str) -> None:
    """A URL the gateway handed back is data: refuse `file:` and friends, which
    `urllib` would otherwise open - a listing naming `file:///etc/shadow` must
    not become a read of this machine's disk."""
    if not isinstance(url, str) or url.split(":", 1)[0].lower() not in ("http", "https"):
        raise UnsafePath(f"refusing a non-HTTP URL: {_redact(str(url))}")


def _redact(url: str) -> str:
    """A presigned URL's query string IS the credential; never put it in an error."""
    return url.split("?", 1)[0]


class _Window:
    """A byte range of an open file, readable like a file - one PUT's body.

    **Reads at least `CHUNK` at a time, whatever it is asked for.** `http.client`
    sends a file-like body in `blocksize` reads, 8 KiB by default, one socket
    write each, and over a ~90 ms round trip that alone held an upload to about
    1 MB/s: 1.1 MB/s measured against 15.8 MB/s with 1 MiB writes (#391 D11).
    `http.client` sends whatever `read` returns, so the size is set here rather
    than on a connection `urllib` builds out of reach."""

    def __init__(self, f, start: int, length: int) -> None:
        self.f, self.left = f, length
        f.seek(start)

    def read(self, n: int = -1) -> bytes:
        if self.left <= 0:
            return b""
        n = self.left if n is None or n < 0 else min(max(n, CHUNK), self.left)
        data = self.f.read(n)
        self.left -= len(data)
        return data


def _when(value) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017


def _expiring(item: Mapping) -> bool:
    """Are this file's URLs within `REFRESH_MARGIN` of their `expires_at`?"""
    expires = _when(item.get("expires_at"))
    return bool(expires and expires - REFRESH_MARGIN <= _now())


class GatewayBackend:
    """Dataset operations against the gateway."""

    writes_listing = False

    def __init__(self, url: str, token: str | None = None, *,
                 transport: Transport | None = None) -> None:
        self.url = url.rstrip("/")
        self._token = token
        self.http = transport or UrllibTransport()

    # --- the API ---------------------------------------------------------------

    def _credential(self) -> str:
        token = self._token or os.environ.get(TOKEN_VARIABLE)
        if not token:
            raise ConfigError(
                f"{self.url} is a dataset gateway and no credential was given: "
                f"{TOKEN_VARIABLE} is unset. Set it to the credential the gateway "
                "issued, or pass `token=`.")
        return token.strip()

    def _call(self, method: str, path: str, payload=None, *, query=None,
              headers: Mapping[str, str] | None = None, timeout: float = API_TIMEOUT,
              on_410: type = Refused) -> dict:
        url = self.url + path + (("?" + urlencode(query)) if query else "")
        sent = {"Authorization": f"Bearer {self._credential()}",
                "Accept": "application/json", **(headers or {})}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            sent["Content-Type"] = "application/json"
        status, raw = self.http.request(method, url, headers=sent, body=body,
                                        timeout=timeout)
        try:
            doc = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, ValueError):
            doc = {}
        if 200 <= status < 300:
            if not isinstance(doc, dict):
                raise BackendUnreachable(f"{method} {path}: the gateway answered non-JSON")
            return doc
        message = (doc.get("error") if isinstance(doc, dict) else None) or (
            raw[:200].decode("utf-8", "replace") or f"HTTP {status}")
        where = f"{method} {path}: {message}"
        if status == 404:
            raise DatasetUnavailable(
                f"{where} (not found, not granted, or an unknown credential)", status=404)
        if status == 410:
            raise on_410(where, status=410)
        if status in (502, 503, 504) or status >= 508 or status == 500:
            raise BackendUnreachable(where, status=status)
        problems = doc.get("problems") if isinstance(doc, dict) else None
        raise Refused(where, status=status, problems=problems)

    # --- reading ---------------------------------------------------------------

    def list(self, prefix: str = "") -> list[DatasetInfo]:
        """Every dataset this credential may read, every page of it."""
        out, after = [], None
        while True:
            query = {"prefix": prefix, "limit": PAGE} | ({"after": after} if after else {})
            page = self._call("GET", "", query=query)
            rows = page.get("datasets")
            if not isinstance(rows, list):
                raise BackendUnreachable("the gateway's listing has no `datasets`")
            for r in rows:
                _layout.check_name(r.get("name"))
                out.append(DatasetInfo(
                    name=r["name"], latest_version=r.get("latest_version"),
                    file_count=r.get("file_count"), total_bytes=r.get("total_bytes"),
                    metadata=r.get("metadata") or {}, moved_to=r.get("moved_to"),
                    published_at=r.get("published_at")))
            if not page.get("truncated"):
                return out
            after = page.get("next")
            if not after:
                raise BackendUnreachable("the gateway said `truncated` with no `next`")

    def resolve(self, name: str, version: int | None = None) -> Version:
        _layout.check_name(name)
        ref = name if version is None else f"{name}@{version}"
        doc = self._call("GET", "/" + ref, on_410=Retracted)
        resolved = _layout.check_name(doc.get("name"))
        got = doc.get("version")
        if not isinstance(got, int) or isinstance(got, bool) or got < 1:
            raise BackendUnreachable(f"{ref}: the gateway answered no version")
        retracted = tuple(r["version"] for r in doc.get("retracted") or []
                          if isinstance(r, Mapping) and isinstance(r.get("version"), int))
        return Version(
            name=resolved, version=got, files=_layout.check_files(doc.get("files")),
            metadata=doc.get("metadata") or {}, requested=name,
            pointers=tuple(doc.get("pointers") or ()), retracted=retracted,
            published_at=doc.get("published_at"))

    def latest(self, name: str) -> int | None:
        try:
            return self.resolve(name).version
        except DatasetUnavailable:
            return None

    def download(self, v: Version, entry: FileEntry, sink: Sink) -> None:
        if not entry.url:
            raise BackendUnreachable(f"{v.name}@{v.version}/{entry.path}: no URL issued")
        status = self.http.stream(entry.url, sink, timeout=TRANSFER_TIMEOUT)
        if status == 403:
            raise UrlExpired(entry.path)
        if status == 404:
            raise DatasetUnavailable(
                f"{v.name}@{v.version}/{entry.path}: the object is gone (retracted "
                "since the resolve?)", status=404)
        if status != 200:
            raise BackendUnreachable(
                f"{v.name}@{v.version}/{entry.path}: object storage answered {status}",
                status=status)

    # --- writing ---------------------------------------------------------------

    def create(self, local_dir: Path, files: tuple[FileEntry, ...], *,
               name: str | None, namespace: str | None, parent: int | None,
               metadata: Mapping | None, label: str | None, breaking: bool = False,
               user_token: str | None = None) -> dict:
        """begin -> PUT each file -> commit.

        An app sends a `namespace` and gets a name assigned under `uploads/`; an
        operator sends a `name` and the `parent` it builds on. `user_token` is
        the identity of the person behind the upload, forwarded as given and
        verified by the gateway; leave it out for a service with no login.
        """
        body: dict = {"files": [{"path": f.path, "size_bytes": f.size_bytes,
                                 "sha256": f.sha256} for f in files]}
        if name is not None:
            body |= {"name": name, "parent": parent, "breaking": breaking}
        if namespace is not None:
            body["namespace"] = namespace
        if label is not None:
            body["label"] = label
        if metadata is not None:
            body["metadata"] = dict(metadata)
        headers = {IDENTITY_HEADER: user_token} if user_token else None
        begun = self._call("POST", "/-/begin", body, headers=headers)
        upload_id = begun.get("upload_id")
        if not isinstance(upload_id, str):
            raise BackendUnreachable("begin answered no upload_id")
        try:
            self._send_all(local_dir, upload_id, begun)
        except BaseException:
            # Interrupted (Ctrl-C included) or given up: release the upload, or it
            # holds its whole declared size against the quota for 24 hours (#497).
            self._abort(upload_id)
            raise
        return self._call("POST", f"/-/commit/{upload_id}", {}, timeout=COMMIT_TIMEOUT)

    def abort(self, upload_id: str) -> dict:
        """Release an open upload this credential began, so its declared size
        stops counting against the quota now rather than in 24 hours."""
        if not (isinstance(upload_id, str) and upload_id.isascii() and upload_id.isalnum()
                and len(upload_id) <= 64):
            raise Refused(f"{upload_id!r} is not an upload id", status=400)
        return self._call("POST", f"/-/abort/{upload_id}", {})

    def _abort(self, upload_id: str) -> None:
        """Best effort, and never the error a caller sees: the upload already
        failed, and an abort that cannot be delivered expires in 24 hours anyway."""
        try:
            self.abort(upload_id)
        except Exception:  # noqa: BLE001, S110
            pass

    def _send_all(self, local_dir: Path, upload_id: str, begun: dict) -> None:
        """Every file, resumable: a file in parts carries on from the part it
        stopped at, on fresh URLs, whether it stopped because its URLs were
        about to expire or because one died under it (#494)."""
        instructions = {f["path"]: f for f in begun.get("files", [])}
        for path in sorted(instructions):
            item = instructions[path]
            if item.get("reuse") or "upload" not in item:
                continue
            _layout.check_path(path)
            source = local_dir.joinpath(*path.split("/"))
            done, refused = 0, 0
            while True:
                if refused or _expiring(item):
                    item = self._fresh(upload_id, path)
                before = done
                done, why = self._send(source, item, done)
                if why is None:
                    break
                if why is _EXPIRING:
                    continue
                # A refusal that came after progress is the first of a new run:
                # what is bounded is refusals in a row, not over a 4 TB file.
                refused = 1 if done > before else refused + 1
                if refused >= SEND_ATTEMPTS:
                    raise BackendUnreachable(
                        f"{path}: object storage refused the upload {SEND_ATTEMPTS} times "
                        f"in a row with fresh URLs (last: {why})")

    def _fresh(self, upload_id: str, path: str) -> dict:
        fresh = self._call("POST", f"/-/urls/{upload_id}", {})
        item = next((f for f in fresh.get("files", []) if f["path"] == path), None)
        if item is None or "upload" not in item:
            raise BackendUnreachable(f"{path}: no fresh upload URL issued")
        return item

    def _send(self, source: Path, item: Mapping, start: int) -> tuple[int, object]:
        """PUT one file, or its parts from part index `start` on. Returns how
        many are done and why it stopped: None when finished, `_EXPIRING` when
        the URLs are about to lapse, else what the dead URL did.

        **A dropped connection is a dead URL, not a dead gateway.** S3 refuses
        an expired presigned PUT as soon as it reads the headers and closes the
        socket while the body is still going up, so the sender sees an aborted
        connection, never the `403` (`WinError 10053`, #494). Asking for fresh
        URLs is right for both readings, and the attempts are bounded."""
        upload = item["upload"]
        if "parts" in upload:
            part_size, parts = upload["part_size"], upload["parts"]
            with open(source, "rb") as f:
                for i in range(start, len(parts)):
                    # Never before the first part of a call: that is what keeps
                    # a skewed clock from refreshing forever without sending.
                    if i > start and _expiring(item):
                        return i, _EXPIRING
                    part = parts[i]
                    offset = (part["part_number"] - 1) * part_size
                    try:
                        status, _ = self.http.request(
                            "PUT", part["url"],
                            headers={"Content-Length": str(part["size_bytes"])},
                            body=_Window(f, offset, part["size_bytes"]),
                            timeout=TRANSFER_TIMEOUT)
                    except BackendUnreachable as exc:
                        return i, f"part {part['part_number']}: {exc}"
                    if status == 403:
                        return i, f"part {part['part_number']}: 403"
                    if status != 200:
                        raise BackendUnreachable(
                            f"{source.name} part {part['part_number']}: object storage "
                            f"answered {status}", status=status)
            return len(parts), None
        with open(source, "rb") as f:
            try:
                status, _ = self.http.request(
                    "PUT", upload["url"], headers=upload["headers"],
                    body=_Window(f, 0, source.stat().st_size),
                    timeout=TRANSFER_TIMEOUT)
            except BackendUnreachable as exc:
                return 0, str(exc)
        if status == 403:
            return 0, "403"
        if status in (200, 412):
            return 1, None
        raise BackendUnreachable(f"{source.name}: object storage answered {status}",
                                 status=status)

    # --- operator actions -----------------------------------------------------

    def move(self, source: str, dest: str, *, breaking: bool = False,
             label: str | None = None) -> dict:
        _layout.parse_ref(source)
        _layout.check_name(dest)
        body = {"source": source, "dest": dest, "breaking": breaking}
        if label:
            body["label"] = label
        return self._call("POST", "/-/move", body, timeout=COMMIT_TIMEOUT)

    def retract(self, name: str, version: int, reason: str) -> dict:
        _layout.check_name(name)
        return self._call("POST", "/-/retract",
                          {"name": name, "version": version, "reason": reason})


def backend(url: str, token: str | None = None, transport: Transport | None = None):
    return GatewayBackend(url, token, transport=transport)
