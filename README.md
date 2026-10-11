# datasets_client

Read shared, versioned datasets from a local folder, a dataset gateway, Azure Blob or S3. One setting, `DATASETS_URL`, picks the backend; the code that reads the data does not change.

## Install

```
pip install "datasets_client @ https://github.com/Nodofact-S-L-U/datasets-client/releases/download/v0.1.0/datasets_client-0.1.0-py3-none-any.whl"
```

Add `[azure]` or `[s3]` for those backends. The base install has no dependencies.

## Use

| `DATASETS_URL` | backend |
|---|---|
| a path, or `file:///path` | a local folder |
| `https://...` | a dataset gateway (credential in `DATASETS_TOKEN`) |
| `az://<account>/<container>/<prefix>` | Azure Blob |
| `s3://<bucket>/<prefix>` | S3 |

```python
from datasets_client import Datasets

ds = Datasets()                                   # reads DATASETS_URL
path = ds.file("example/dataset01", "data.csv")   # latest version, cached locally
```

Every file is checked against its sha256 before it is returned. A failure raises a named error; it never returns an empty result.

## License

MIT
