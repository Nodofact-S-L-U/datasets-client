"""`record`: where the run executes. Writes `run.json` into a local run folder.

```python
from datasets_client.evals import begin
run = begin("runs/", suite="tenday", suites="evals/suites.json", engine="eng",
            engine_version="eng-2026-10-22-1", kind="ci", reason="nightly",
            inputs=[{"role": "scored", "name": "example/dataset02"}],
            params={"min_nodes": 3})
...                                    # the demo's harness runs and scores
run.finish(metrics={"all": {"recall": 0.91}}, scorer={...}, gate=None)
```

`begin` assigns the run id, resolves every input with whatever credential is
present (on a compute box, a read-only CI token), computes each digest and
probes the machine; `finish` probes it again and writes the validated record.
The library never computes a metric, a scorer or a verdict: the harness hands
them over. `record(...)` is both in one call, for a harness that kept its own
start time.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from datasets_client import _layout
from datasets_client._client import Datasets
from datasets_client.errors import Refused
from datasets_client.evals import _core
from datasets_client.evals._core import CLIENT_REFUSAL, EvalError

RUN_FILE = "run.json"
COLLISION_RETRIES = 5


# --- the machine ------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _cpu_model() -> str:
    info = _read("/proc/cpuinfo")
    if info:
        for line in info.splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"


def _cores(threads: int | None) -> int | None:
    """Physical cores where the system says; else the logical count."""
    info = _read("/proc/cpuinfo")
    if info:
        seen, phys = set(), None
        for line in info.splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "physical id":
                phys = value.strip()
            elif key == "core id":
                seen.add((phys, value.strip()))
        if seen:
            return len(seen)
    return threads


def _memory_bytes() -> int | None:
    info = _read("/proc/meminfo")
    if info:
        for line in info.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    if hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    if os.name == "nt":
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("pagefile", ctypes.c_ulonglong), ("availpage", ctypes.c_ulonglong),
                            ("virtual", ctypes.c_ulonglong), ("availvirt", ctypes.c_ulonglong),
                            ("ext", ctypes.c_ulonglong)]

            status = _Status()
            status.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total)
        except (AttributeError, OSError):
            pass
    return None


def _cpu_times() -> tuple[int, int] | None:
    """(busy, total) since boot, in whatever unit the system counts."""
    stat = _read("/proc/stat")
    if stat and stat.startswith("cpu "):
        values = [int(v) for v in stat.splitlines()[0].split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values) - idle, sum(values)
    if os.name == "nt":
        try:
            import ctypes

            idle, kernel, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(),
                                  ctypes.c_ulonglong())
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                                     ctypes.byref(user)):
                total = kernel.value + user.value      # kernel time includes idle
                return total - idle.value, total
        except (AttributeError, OSError):
            pass
    return None


def load_probe(sample_s: float = 0.2) -> dict:
    """`{loadavg_1m, cpu_busy_pct}`. Either is null where the system cannot say
    (Windows has no load average). A rough gauge, and named as one."""
    try:
        loadavg = round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        loadavg = None
    busy = None
    first = _cpu_times()
    if first is not None:
        time.sleep(sample_s)
        second = _cpu_times()
        if second is not None and second[1] > first[1]:
            busy = round(100.0 * (second[0] - first[0]) / (second[1] - first[1]), 1)
    return {"loadavg_1m": loadavg, "cpu_busy_pct": busy}


def _peak_rss() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    # Kilobytes on Linux, bytes on macOS.
    return int(peak) if platform.system() == "Darwin" else int(peak) * 1024


def _env_class() -> str:
    if os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted":
        return "gh-hosted"
    if os.environ.get("SLURM_JOB_ID"):
        return "compute-box"
    return "laptop"


def _allocation(exclusive: bool | None) -> dict | None:
    if not os.environ.get("SLURM_JOB_ID"):
        return None
    cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if cpus is None and hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    mem = os.environ.get("SLURM_MEM_PER_NODE")
    return {"cpus": int(cpus) if cpus is not None and str(cpus).isdigit() else None,
            "memory_bytes": int(mem) * 1024 * 1024 if mem and mem.isdigit() else None,
            "exclusive": bool(exclusive)}


def machine(*, env_class: str | None = None, resource: str | None = None,
            exclusive: bool | None = None) -> dict:
    """`env` without `load`: what the run executed on."""
    threads = os.cpu_count()
    cls = env_class or _env_class()
    if cls not in _core.ENV_CLASSES:
        raise EvalError(f"env class {cls!r} is not one of {', '.join(_core.ENV_CLASSES)}")
    return {"class": cls, "resource": resource,
            "host": None if resource else socket.gethostname().lower(),
            "cpu_model": _cpu_model(), "cores": _cores(threads), "threads": threads,
            "memory_bytes": _memory_bytes(),
            "os": f"{platform.system()} {platform.release()}".strip(),
            "python": platform.python_version(), "allocation": _allocation(exclusive),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


# --- where it ran from ------------------------------------------------------------


def trigger() -> dict | None:
    """From the `GITHUB_*` variables and the event payload; None outside Actions."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None
    ref = os.environ.get("GITHUB_REF")
    default = None
    payload = _read(os.environ.get("GITHUB_EVENT_PATH", "")) if os.environ.get(
        "GITHUB_EVENT_PATH") else None
    if payload:
        try:
            branch = (json.loads(payload).get("repository") or {}).get("default_branch")
        except (ValueError, AttributeError):
            branch = None
        if isinstance(branch, str) and ref:
            default = ref == f"refs/heads/{branch}"
    if default is None and os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        # A `schedule` payload carries no `repository` object, and a scheduled
        # workflow runs only on the default branch, which `GITHUB_REF` names.
        default = bool(ref)
    return {"event": os.environ.get("GITHUB_EVENT_NAME"), "ref": ref,
            "default_branch": default, "repository": os.environ.get("GITHUB_REPOSITORY"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}


def _gh_run() -> tuple[str, str] | None:
    rid, attempt = os.environ.get("GITHUB_RUN_ID"), os.environ.get("GITHUB_RUN_ATTEMPT")
    if (os.environ.get("GITHUB_ACTIONS") == "true" and rid and attempt and rid.isdigit()
            and attempt.isdigit() and len(rid) <= 12 and len(attempt) <= 3):
        return rid, attempt
    return None


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,  # noqa: S603,S607
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_facts(repo: str | os.PathLike = ".") -> dict:
    """`{commit, commit_time, dirty}` from git, each null when it cannot be known.
    `dirty` ignores untracked files: a run folder inside the repository is one."""
    root = Path(repo)
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None or len(commit) != 40:
        return {"commit": None, "commit_time": None, "dirty": None}
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    when = _core.parse_time(_git(root, "show", "-s", "--format=%cI", commit))
    return {"commit": commit.lower(), "commit_time": _core.iso(when) if when else None,
            "dirty": None if status is None else bool(status)}


# --- inputs -----------------------------------------------------------------------


def resolve_inputs(ds: Datasets, inputs: Sequence[Mapping]) -> list[dict]:
    """Each input resolved at the version read, digested, classified. A client
    input is refused before anything is written."""
    out = []
    for spec in inputs:
        if not isinstance(spec, Mapping):
            raise EvalError("each input is {role, name[, version][, path]}")
        name, version = _layout.parse_ref(spec.get("name"))
        version = spec.get("version", version)
        v = ds.resolve(name, version=version)
        meta = v.metadata or {}
        cls = _layout.data_class_of(meta)
        if cls == "client":
            raise Refused(f"{name}: {CLIENT_REFUSAL}", status=403)
        if cls is None:
            raise EvalError(f"{name}@{v.version} states no data_class")
        path = spec.get("path")
        entry = {"role": spec.get("role"), "name": v.name, "version": v.version,
                 "path": path, "digest": _core.content_digest(v.files, path),
                 "data_class": cls, "sealed": meta.get("purpose") == "sealed"}
        if cls == "public":
            entry["licence"] = meta.get("licence")
        out.append(entry)
    return out


def load_suite(suites: str | os.PathLike | Mapping, name: str) -> dict:
    """One suite's entry from the demo's own `suites.json` (or its parsed
    document)."""
    if isinstance(suites, Mapping):
        doc = suites
    else:
        try:
            doc = json.loads(Path(suites).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EvalError(f"cannot read the suites file {suites}: {exc}") from None
    doc = _core.check_suites_doc(doc)
    if name not in doc["suites"]:
        raise EvalError(f"suite {name!r} is not in the suites file; it has "
                        f"{', '.join(sorted(doc['suites'])) or '(none)'}")
    return doc["suites"][name]


# --- the run folder ---------------------------------------------------------------


class RunFolder:
    """One run being recorded. `finish` writes `run.json`; `progress` notes how
    far a long run got, so a run stopped early is still a record."""

    def __init__(self, path: Path, doc: dict, suite: Mapping) -> None:
        self.path = path
        self.doc = doc
        self.suite = suite

    @property
    def run_id(self) -> str:
        return self.doc["run_id"]

    def progress(self, phase: str, done: int, total: int | None = None) -> None:
        self.doc["progress"].append({"phase": phase, "done": done, "total": total,
                                     "at": _core.iso(_core.now())})

    def finish(self, *, metrics: Mapping, scorer: Mapping, gate: Mapping | None = None,
               status: str = "complete", notes: str | None = None,
               timings: Mapping | None = None, peak_rss_bytes: int | None = None,
               load_end: Mapping | None = None, ended_at: str | None = None,
               rescored_from: str | None = None, extra: Mapping | None = None) -> Path:
        """Validate and write `run.json`. `extra` is the demo's own top-level
        keys, kept and ignored by every query - except `publication`, which only
        `publish` writes (§4.1): one a demo wrote would be trusted as its own."""
        if extra and "publication" in extra:
            raise EvalError("run.json: publication is written by publish, never by the demo")
        doc = dict(extra or {}) | self.doc
        doc |= {"metrics": dict(metrics), "scorer": dict(scorer),
                "gate": dict(gate) if gate is not None else None, "status": status,
                "notes": notes, "rescored_from": rescored_from,
                "timings": dict(timings) if timings is not None else None,
                "ended_at": ended_at or _core.iso(_core.now())}
        doc["env"] = dict(doc["env"]) | {
            "load": {"start": doc["env"]["load"]["start"],
                     "end": dict(load_end) if load_end is not None else load_probe()},
            "peak_rss_bytes": peak_rss_bytes if peak_rss_bytes is not None else _peak_rss()}
        _core.check_run(doc, suite=self.suite)
        raw = (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
        _core.run_size_ok(raw)
        tmp = self.path / f".{RUN_FILE}.part"
        tmp.write_bytes(raw)
        os.replace(tmp, self.path / RUN_FILE)
        self.doc = doc
        return self.path


def begin(out_root: str | os.PathLike, *, suite: str, suites: str | os.PathLike | Mapping,
          engine: str, engine_version: str, kind: str, reason: str,
          inputs: Sequence[Mapping], params: Mapping, variant: str | None = None,
          experiment: str | None = None, commit: str | None | object = ...,
          dirty: bool | None | object = ..., commit_time: str | None | object = ...,
          repo: str | os.PathLike = ".", env_class: str | None = None,
          resource: str | None = None, exclusive: bool | None = None,
          ds: Datasets | None = None, started_at: datetime | None = None,
          load_start: Mapping | None = None, input_inferred: bool = False,
          prepared_inputs: Sequence[Mapping] | None = None) -> RunFolder:
    """Start a run: its id, its folder `<out_root>/<run_id>/`, its inputs and
    its machine. Commit, dirty and commit time come from git in `repo` unless
    given. `prepared_inputs` is for a backfill whose inputs were reconstructed
    (with `input_inferred=True`); everything else resolves them."""
    _core.check_suite(suite)
    _core.check_part(engine, "engine", _core.MAX_ENGINE)
    if variant is not None:
        _core.check_part(variant, "variant", _core.MAX_VARIANT)
    entry = load_suite(suites, suite)
    facts = git_facts(repo) if ... in (commit, dirty, commit_time) else {}
    commit = facts.get("commit") if commit is ... else commit
    dirty = facts.get("dirty") if dirty is ... else dirty
    commit_time = facts.get("commit_time") if commit_time is ... else commit_time
    if isinstance(commit, str):
        commit = commit.lower()
    if prepared_inputs is not None:
        resolved = [dict(i) for i in prepared_inputs]
    else:
        resolved = resolve_inputs(ds or Datasets(), inputs)
    for i in resolved:
        if i.get("data_class") == "client":
            raise Refused(f"{i.get('name')}: {CLIENT_REFUSAL}", status=403)
    _sealed_declared(entry, suite, resolved)
    scored = [i for i in resolved if i.get("role") == "scored"]
    if len(scored) != 1:
        raise EvalError("exactly one input has role 'scored'")
    if scored[0]["name"] != _core.scored_input(entry):
        raise EvalError(f"suite {suite} scores {_core.scored_input(entry)}, not "
                        f"{scored[0]['name']}")
    env = machine(env_class=env_class, resource=resource, exclusive=exclusive)
    start_probe = dict(load_start) if load_start is not None else load_probe()
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    gh = _gh_run()
    for attempt in range(COLLISION_RETRIES + 1):
        when = started_at or _core.now()
        run_id = _core.make_run_id(when, engine, commit, gh)
        folder = root / run_id
        try:
            folder.mkdir()
            break
        except FileExistsError:
            # A sweep: the same engine and commit in the same second. The id
            # names the second, so wait one and take the next.
            if attempt == COLLISION_RETRIES or started_at is not None:
                raise EvalError(f"run id {run_id} is already recorded in {root}") from None
            time.sleep(1)
    doc = {
        "schema": _core.RUN_SCHEMA, "run_id": run_id,
        "suite": {"name": suite, "digest": _core.suite_digest(entry)},
        "subject": {"engine": engine, "engine_version": engine_version, "commit": commit,
                    "commit_time": commit_time, "dirty": dirty, "variant": variant,
                    "config": {"params": dict(params)},
                    "config_digest": _core.config_digest(params)},
        "inputs": resolved, "input_inferred": bool(input_inferred),
        "reason": reason, "experiment": experiment, "kind": kind, "progress": [],
        "trigger": trigger(), "env": env | {"load": {"start": start_probe}},
        "started_at": _core.iso(when),
    }
    return RunFolder(folder, doc, entry)


def _sealed_declared(entry: Mapping, suite: str, inputs: Sequence[Mapping]) -> None:
    """A suite that names answer-key fields reads a hold-out: refuse a run of it
    whose inputs none say `purpose: sealed`. Stripping keys on the marker, a
    missing marker would publish the answer key, so this fails closed."""
    if entry.get("sealed_fields") and not any(i.get("sealed") for i in inputs):
        raise EvalError(f"suite {suite} declares sealed_fields, and no input of this run is "
                        "sealed: mark the hold-out dataset `purpose: sealed` in its metadata "
                        "first")


def record(out_root: str | os.PathLike, *, metrics: Mapping, scorer: Mapping,
           gate: Mapping | None = None, status: str = "complete", notes: str | None = None,
           timings: Mapping | None = None, peak_rss_bytes: int | None = None,
           load_end: Mapping | None = None, ended_at: str | None = None,
           rescored_from: str | None = None, extra: Mapping | None = None,
           **start) -> Path:
    """`begin` and `finish` in one call; returns the run folder."""
    run = begin(out_root, **start)
    try:
        return run.finish(metrics=metrics, scorer=scorer, gate=gate, status=status,
                          notes=notes, timings=timings, peak_rss_bytes=peak_rss_bytes,
                          load_end=load_end, ended_at=ended_at, rescored_from=rescored_from,
                          extra=extra)
    except BaseException:
        # A record that did not validate leaves no folder to be published.
        if not (run.path / RUN_FILE).exists():
            try:
                run.path.rmdir()
            except OSError:
                pass
        raise


def read_run(folder: str | os.PathLike) -> dict:
    path = Path(folder) / RUN_FILE
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvalError(f"cannot read {path}: {exc}") from None
    _core.run_size_ok(raw)
    try:
        doc = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise EvalError(f"{path} is not JSON") from None
    return doc
