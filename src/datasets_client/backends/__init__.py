"""Which backend a URL means. The scheme decides, and nothing else does.

| URL | backend |
|---|---|
| `file:///path`, or a bare path | a local folder holding the layout |
| `http://...`, `https://...` | a dataset gateway's datasets API (`DATASETS_TOKEN`) |
| `az://<account>/<container>/<prefix>` | Azure Blob Storage (`[azure]` extra) |
| `s3://<bucket>/<prefix>` | S3 with the ambient AWS identity (`[s3]` extra) |
"""

from __future__ import annotations

import re

from datasets_client.errors import ConfigError

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def open_backend(url: str, *, token: str | None = None, transport=None):
    if not isinstance(url, str) or not url.strip():
        raise ConfigError("DATASETS_URL is not set: name a folder, a gateway URL, "
                          "az://<account>/<container> or s3://<bucket>")
    url = url.strip()
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme in ("", "file") or _WINDOWS_DRIVE.match(url):
        from datasets_client.backends import file
        return file.backend(url)
    if scheme in ("http", "https"):
        from datasets_client.backends import http_gateway
        return http_gateway.backend(url, token, transport)
    if scheme == "az":
        from datasets_client.backends import azure
        return azure.backend(url)
    if scheme == "s3":
        from datasets_client.backends import s3
        return s3.backend(url)
    raise ConfigError(f"{url}: unknown scheme {scheme!r}; expected a folder, http(s), "
                      "az:// or s3://")
