"""The named errors. Every way a read can fail has one, and none of them is `[]`.

The rule the whole library is built around: **"I could not read it" never looks
like "there is nothing there".** A backend that cannot list raises; a dataset
that is not granted raises; a file whose bytes do not match its listing raises
and leaves nothing behind. An application can catch `DatasetError` to handle
all of them at once, or one of the subclasses to tell them apart.
"""

from __future__ import annotations


class DatasetError(Exception):
    """Base class. `status` is the HTTP-style status when the backend gave one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class ConfigError(DatasetError):
    """`DATASETS_URL` is unset or unusable, an optional extra is not installed,
    or a credential the backend needs is missing. Raised when a dataset is
    READ, never at import, so an app that reads no dataset still starts."""


class DatasetUnavailable(DatasetError):
    """The dataset does not exist, **or this caller may not read it** - one
    error, because the gateway gives one answer for both so that another space
    cannot be enumerated. Also a file the dataset's listing does not contain."""


class BackendUnreachable(DatasetError):
    """The backend could not be asked: a network failure, a 5xx, a folder that
    does not exist, a container the identity cannot list. The only error
    `allow_stale=True` falls back on."""


class IntegrityError(DatasetError):
    """A downloaded file's size or sha256 differs from the listing. The bytes
    were discarded; nothing was written where a reader could find it."""


class Retracted(DatasetError):
    """The version asked for was retracted. It is never served again."""


class UnsafePath(DatasetError):
    """A listing named a path that could escape the cache or alias a file:
    `..`, an absolute path, a device name, or one that resolves outside the
    cache root. Raised before a byte is written."""


class Refused(DatasetError):
    """The backend understood and said no: a bad request, a stale parent
    (`409`), a quota (`507`), a commit whose objects did not verify (`422`).
    `problems` carries the per-file reasons of a `422`."""

    def __init__(self, message: str, *, status: int | None = None,
                 problems: list[dict] | None = None) -> None:
        super().__init__(message, status=status)
        self.problems = problems or []
