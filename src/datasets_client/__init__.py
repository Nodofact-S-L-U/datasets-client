"""datasets_client - read shared datasets from wherever they live.

One setting, `DATASETS_URL`, and its scheme picks the backend: a local folder,
a dataset gateway (`http(s)://`), Azure Blob (`az://`) or S3 (`s3://`). Product code names none
of them; the same image runs anywhere with only that value changed.

```python
from datasets_client import Datasets
ds = Datasets()
path = ds.file("example/dataset01", "data.csv")   # a real local path
```

Every file is checked against its sha256 before it is returned, and a backend
that cannot be read raises a named error - never an empty result.
"""

from datasets_client._client import Datasets, Fetched, copy
from datasets_client._layout import DatasetInfo, FileEntry, Version
from datasets_client.errors import (
    BackendUnreachable,
    ConfigError,
    DatasetError,
    DatasetUnavailable,
    IntegrityError,
    Refused,
    Retracted,
    UnsafePath,
)

__version__ = "0.1.0"

__all__ = [
    "BackendUnreachable", "ConfigError", "DatasetError", "DatasetInfo",
    "DatasetUnavailable", "Datasets", "Fetched", "FileEntry", "IntegrityError",
    "Refused", "Retracted", "UnsafePath", "Version", "__version__", "copy",
]
