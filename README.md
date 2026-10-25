# datasets_client

Read shared, versioned datasets from a local folder, a dataset gateway, Azure Blob or S3. One setting, `DATASETS_URL`, picks the backend; the code that reads the data does not change.

## Install

```
pip install "datasets_client @ https://github.com/Nodofact-S-L-U/datasets-client/releases/download/v0.3.2/datasets_client-0.3.2-py3-none-any.whl"
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

A version's files are fetched 8 at a time (`DATASETS_DOWNLOAD_THREADS`, or `Datasets(download_threads=...)`), and a transfer the remote resets is retried. `pull` reports its progress on stderr.

## Evaluation runs

`datasets_client.evals` stores an evaluation run (`run.json`, schema `eval-run/1`) as an ordinary dataset, keeps a small append-only registry of suites, metrics, pins, notes and contamination records beside the runs, and answers questions about them: `scoreboard`, `best`, `history`, `compare`, `diff`, `show`, `experiments`, `search`, `coverage`, `regressions` and `catalog`.

```
python -m datasets_client evals --help-json      # every verb and its arguments
python -m datasets_client evals scoreboard       # answers are JSON; add --table for people
```

It works on every backend, including a local folder (`DATASETS_URL=file://...`).

## License

MIT
