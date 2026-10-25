"""`datasets_client.evals` - evaluation runs, stored as datasets, and queried.

A demo runs its own evaluations and computes its own verdict. This module:

* **records** a run where it executes (`begin` / `finish`, or `record`), as a
  folder holding `run.json` (`eval-run/1`) and optional `artifacts/`;
* **publishes** it as the dataset `<ns>-evals/runs/<suite>/<run-id>`, version
  1, never versioned (`Evals.publish`);
* **registers** what the demo declares - suites, metrics, pins, notes and
  contamination records - under `<ns>-evals/registry/g<N>/` (`Evals.init`,
  `Evals.pin`, ...);
* **answers queries** over all of it as JSON (`scoreboard`, `best`, `history`,
  `compare`, `diff`, `show`, `experiments`, `experiment`, `search`, `coverage`,
  `regressions`, `catalog`).

```python
from datasets_client.evals import Evals, scoreboard
ev = Evals("example-evals")                  # reads DATASETS_URL
ev.publish("runs/20261024-213000-eng-3f9a2c1")
print(scoreboard(ev)["ranking"])
```

**It never scores.** It stores and compares numbers the demo produced; it never
recomputes a metric and never re-gates a run. On the command line:
`python -m datasets_client evals <verb> ...`, JSON by default.
"""

from datasets_client.evals._core import (
    EvalError,
    canonical,
    config_digest,
    content_digest,
    scorer_digest,
    strip_sealed,
)
from datasets_client.evals._query import (
    best,
    catalog,
    compare,
    coverage,
    diff,
    experiment,
    experiments,
    help_json,
    history,
    regressions,
    scoreboard,
    search,
    show,
)
from datasets_client.evals._record import RunFolder, begin, git_facts, load_probe, record
from datasets_client.evals._store import Evals, Registry, Run, unpinnable

__all__ = [
    "EvalError", "Evals", "Registry", "Run", "RunFolder", "begin", "best", "canonical",
    "catalog", "compare", "config_digest", "content_digest", "coverage", "diff",
    "experiment", "experiments", "git_facts", "help_json", "history", "load_probe",
    "record", "regressions", "scoreboard", "scorer_digest", "search", "show",
    "strip_sealed", "unpinnable",
]
