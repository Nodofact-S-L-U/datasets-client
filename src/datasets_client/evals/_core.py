"""The shapes: names, digests, `run.json` (`eval-run/1`), the registry files and
a run's dataset metadata. Pure functions, no backend.

Everything here is a check the library makes on the machine that writes, so a
run that would not answer a query is refused when it is recorded, not when
somebody asks about it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from datasets_client import _layout
from datasets_client.errors import DatasetError, Refused

RUN_SCHEMA = "eval-run/1"
SUITES_SCHEMA = "eval-suites/1"
METRICS_SCHEMA = "eval-metrics/1"
QUERY_SCHEMA = "eval-query/1"
SUFFIX = "-evals"
MAX_BASE = 58
MAX_SUITE = 40
MAX_ENGINE = 20
MAX_VARIANT = 32
MAX_RUN_JSON = 1024 * 1024
MAX_PARAMS = 64 * 1024
MAX_NOTES = 64 * 1024
MAX_NOTE_TEXT = 8 * 1024
MAX_REASON = 200
MAX_RUN_METADATA = 1024
MAX_REGISTRY_METADATA = 16 * 1024
HEADLINE = 4

KINDS = ("release", "ci", "pr", "experiment", "manual", "rescore")
STATUSES = ("complete", "partial", "failed")
ROLES = ("scored", "calibration", "truth", "topology")
VERDICTS = ("pass", "fail", "incomparable")
ENV_CLASSES = ("compute-box", "bench", "gh-hosted", "laptop")
METRIC_KINDS = ("rate", "count", "time", "context")
NOTE_TAGS = ("invalid", "conclusion", "hypothesis")
CONTAMINATION_KINDS = ("development_informed", "spent_holdout")
REGISTRY = ("suites", "metrics", "pins", "notes", "contamination")
REGISTRY_FILES = {"suites": "suites.json", "metrics": "metrics.json",
                  "pins": "pins.jsonl", "notes": "notes.jsonl",
                  "contamination": "contamination.jsonl"}
CLIENT_REFUSAL = ("evaluation runs on client data are deferred (#11 6388991117, "
                  "decision 5)")

_PART = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STAMP = re.compile(r"^\d{8}-\d{6}$")
_CADENCE = re.compile(r"^(P[1-9][0-9]{0,3}D|PT[1-9][0-9]{0,4}H)$")
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LINE_ID = re.compile(r"^[pnc]-\d{8}-\d{6}-[0-9a-f]{4}$")


class EvalError(DatasetError):
    """A request the evaluation layer cannot answer as asked: an unknown
    metric, a refused pin, a record that does not validate. The message says
    what would be valid instead."""

    def __init__(self, message: str, *, status: int | None = 400) -> None:
        super().__init__(message, status=status)


# --- time -------------------------------------------------------------------------


def now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - the package supports 3.10


def iso(when: datetime) -> str:
    """UTC, whole seconds, with a `Z`."""
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017


def parse_time(value: object) -> datetime | None:
    """`...Z` or any ISO 8601 offset; None for anything else."""
    if not isinstance(value, str) or not value:
        return None
    try:
        got = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return got if got.tzinfo else got.replace(tzinfo=timezone.utc)  # noqa: UP017


# --- names ------------------------------------------------------------------------


def check_part(value: object, what: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit or not _PART.match(value):
        raise EvalError(f"{what} {value!r}: lower-case letters, digits, '.', '_' and '-', "
                        f"starting with a letter or a digit, at most {limit} characters")
    return value


def check_suite(value: object) -> str:
    check_part(value, "suite", MAX_SUITE)
    if value == "registry":
        raise EvalError("suite 'registry' is reserved")
    return value


def check_evals(ns: object) -> str:
    """`<ns>-evals`, `<ns>` at most 58 characters."""
    if (not isinstance(ns, str) or len(ns) > _layout.MAX_SEGMENT or not _PART.match(ns)
            or not ns.endswith(SUFFIX) or not 1 <= len(ns) - len(SUFFIX) <= MAX_BASE):
        raise EvalError(f"{ns!r} is not an evaluation namespace: it is `<ns>-evals`, "
                        f"with `<ns>` at most {MAX_BASE} characters")
    return ns


def run_name(evals: str, suite: str, run_id: str) -> str:
    return _layout.check_name(f"{evals}/runs/{suite}/{run_id}")


def registry_name(evals: str, generation: int, kind: str) -> str:
    return _layout.check_name(f"{evals}/registry/g{generation}/{kind}")


def subject_of(engine: str, variant: str | None) -> str:
    return engine if variant is None else f"{engine}:{variant}"


def sha7(commit: str | None) -> str:
    return commit[:7] if commit else "unknown"


def make_run_id(started: datetime, engine: str, commit: str | None,
                gh_run: tuple[str, str] | None) -> str:
    stamp = started.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")  # noqa: UP017
    rid = f"{stamp}-{engine}-{sha7(commit)}"
    if gh_run:
        rid += f"-{gh_run[0]}-{gh_run[1]}"
    return check_run_id(rid.lower(), engine=engine, commit=commit)


def check_run_id(rid: object, *, engine: str | None = None,
                 commit: str | None | object = ...) -> str:
    """`<YYYYMMDD-hhmmss>-<engine>-<sha7>[-<gh-run-id>-<attempt>]`, and - given
    the engine and commit - the id of THAT engine and commit."""
    if not isinstance(rid, str) or len(rid) > 61 or rid != rid.lower():
        raise EvalError(f"run id {rid!r}: lower case, at most 61 characters")
    eng = re.escape(engine) if engine is not None else r"[a-z0-9][a-z0-9._-]{0,19}"
    sha = (re.escape(sha7(commit)) if commit is not ... else r"(?:[0-9a-f]{7}|unknown)")
    if not re.match(rf"^\d{{8}}-\d{{6}}-{eng}-{sha}(?:-\d{{1,12}}-\d{{1,3}})?$", rid):
        raise EvalError(f"run id {rid!r} is not <YYYYMMDD-hhmmss>-<engine>-<sha7>"
                        "[-<gh-run-id>-<attempt>]" + (f" for engine {engine!r}" if engine
                                                       else ""))
    try:
        datetime.strptime(rid[:15], "%Y%m%d-%H%M%S")  # noqa: DTZ007 - a format check
    except ValueError:
        raise EvalError(f"run id {rid!r} does not start with a real UTC time") from None
    return rid


# --- canonical JSON and digests ---------------------------------------------------


def canonical(value) -> bytes:
    """The one serialisation every digest is over (§5)."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise EvalError(f"not canonical JSON: {exc}") from None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pairs_digest(pairs: Iterable[tuple[str, str]]) -> str:
    return sha256_hex(canonical(sorted([p, h] for p, h in pairs)))


def content_digest(files: Iterable, path: str | None = None) -> str:
    """§5.1: over a version's listing, or the sub-tree `path` of it."""
    chosen = [(f.path, f.sha256) for f in files
              if path is None or f.path == path or f.path.startswith(path + "/")]
    if not chosen:
        raise EvalError(f"no file at or under {path!r} in that version")
    return pairs_digest(chosen)


def config_digest(params: Mapping) -> str:
    return sha256_hex(canonical(dict(params)))


CRLF, LF = bytes([13, 10]), bytes([10])


def scorer_digest(root: str | Path, paths: Iterable[str]) -> str:
    """§5.3: the scorer's source files, by their paths relative to `root`. The
    demo chooses the paths; the library only hashes them, with CRLF read as LF so
    a Windows checkout under `core.autocrlf` gives the committed file's digest."""
    base = Path(root)
    pairs = []
    for p in paths:
        rel = str(p).replace("\\", "/").strip("/")
        target = base.joinpath(*rel.split("/"))
        if not target.is_file():
            raise EvalError(f"scorer path {rel!r} is not a file under {base}")
        pairs.append((rel, sha256_hex(target.read_bytes().replace(CRLF, LF))))
    if not pairs:
        raise EvalError("a scorer digest needs at least one path")
    return pairs_digest(pairs)


def suite_digest(entry: Mapping) -> str:
    return sha256_hex(canonical(dict(entry)))


def sig6(value):
    """A number rounded to 6 significant digits; what a tie and a headline use."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return value
    if value == 0 or not math.isfinite(value):
        return value
    return float(f"{value:.6g}")


# --- the registry documents -------------------------------------------------------


def check_metrics_doc(doc: object) -> dict:
    """`eval-metrics/1`: name, path, better, unit, kind, denominator."""
    if not isinstance(doc, Mapping) or doc.get("schema") != METRICS_SCHEMA:
        raise EvalError(f"a metrics file is an object with schema {METRICS_SCHEMA!r}")
    metrics = doc.get("metrics")
    if not isinstance(metrics, list):
        raise EvalError("metrics.metrics is a list")
    names: dict[str, dict] = {}
    for m in metrics:
        if not isinstance(m, Mapping):
            raise EvalError("each metric is an object")
        name = check_part(m.get("name"), "metric", MAX_SUITE)
        if name in names:
            raise EvalError(f"metric {name!r} is declared twice")
        if "better" not in m or m["better"] not in ("up", "down", None):
            raise EvalError(f"metric {name}: better is 'up', 'down', or null for a metric "
                            "with no right direction (shown and compared, never ranked)")
        if m.get("kind") not in METRIC_KINDS:
            raise EvalError(f"metric {name}: kind is one of {', '.join(METRIC_KINDS)}")
        for key in ("path", "unit"):
            if not isinstance(m.get(key), str):
                raise EvalError(f"metric {name}: {key} is a string")
        names[name] = dict(m)
    for name, m in names.items():
        den = m.get("denominator")
        if m["kind"] == "count":
            if den is not None and (den not in names or den == name):
                raise EvalError(f"metric {name} is a count: denominator is null or names "
                                "another declared metric holding the total")
        elif den is not None:
            raise EvalError(f"metric {name}: only a count has a denominator")
    return dict(doc)


def metric_table(doc: Mapping | None) -> dict[str, dict]:
    return {m["name"]: m for m in (doc or {}).get("metrics", [])}


def check_suites_doc(doc: object, metrics: Mapping[str, Mapping] | None = None) -> dict:
    """`eval-suites/1`. With `metrics`, every primary and tie-break must be one."""
    if not isinstance(doc, Mapping) or doc.get("schema") != SUITES_SCHEMA:
        raise EvalError(f"a suites file is an object with schema {SUITES_SCHEMA!r}")
    suites = doc.get("suites")
    if not isinstance(suites, Mapping):
        raise EvalError("suites.suites is an object keyed by suite name")
    for name, s in suites.items():
        check_suite(name)
        if not isinstance(s, Mapping):
            raise EvalError(f"suite {name} is an object")
        inputs = s.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise EvalError(f"suite {name}: inputs is a non-empty list")
        for i in inputs:
            if not isinstance(i, Mapping) or i.get("role") not in ROLES:
                raise EvalError(f"suite {name}: each input has a role, one of "
                                f"{', '.join(ROLES)}")
            _layout.check_name(i.get("name"))
            if "@" in i["name"]:
                raise EvalError(f"suite {name}: an input names a dataset without a version")
        if sum(1 for i in inputs if i["role"] == "scored") != 1:
            raise EvalError(f"suite {name}: exactly one input is scored")
        views = s.get("views")
        if not isinstance(views, Mapping) or "all" not in views:
            raise EvalError(f"suite {name}: views is an object and declares 'all'")
        for v in views:
            check_part(v, "view", _layout.MAX_SEGMENT)
        primary, tie = s.get("primary"), s.get("tie_break", [])
        if not isinstance(tie, list):
            raise EvalError(f"suite {name}: tie_break is a list")
        for m in [primary, *tie]:
            check_part(m, f"suite {name} metric", MAX_SUITE)
            if metrics is not None and m not in metrics:
                raise EvalError(f"suite {name} names metric {m!r}, which is not registered; "
                                f"registered: {', '.join(sorted(metrics)) or '(none)'}")
            if metrics is not None and metrics[m].get("better") is None:
                raise EvalError(f"suite {name} ranks by metric {m!r}, which has no "
                                "direction (better: null)")
        cadence = s.get("cadence")
        if cadence is not None and (not isinstance(cadence, str) or not _CADENCE.match(cadence)):
            raise EvalError(f"suite {name}: cadence is P<n>D, PT<n>H or null")
        family = s.get("family")
        if family is not None:
            check_part(family, f"suite {name} family", MAX_SUITE)
        sealed = s.get("sealed_fields")
        if sealed is not None and (not isinstance(sealed, list) or not all(
                isinstance(f, str) and f for f in sealed)):
            raise EvalError(f"suite {name}: sealed_fields is a list of key names")
        clash = sorted(set(sealed or ()) & set(metrics or ()))
        if clash:
            raise EvalError(f"suite {name}: sealed_fields names metric {', '.join(clash)}; a "
                            "sealed field is stripped at any depth, so rename the metric")
    return dict(doc)


def scored_input(suite: Mapping) -> str:
    return next(i["name"] for i in suite["inputs"] if i["role"] == "scored")


def cadence_map(doc: Mapping) -> dict[str, str]:
    return {name: s["cadence"] for name, s in sorted(doc.get("suites", {}).items())
            if s.get("cadence")}


def parse_jsonl(raw: bytes, what: str) -> list[dict]:
    out = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            raise EvalError(f"{what} line {n} is not JSON") from None
        if not isinstance(doc, dict):
            raise EvalError(f"{what} line {n} is not an object")
        out.append(doc)
    return out


def jsonl_line(doc: Mapping) -> bytes:
    return canonical(dict(doc)) + b"\n"


# --- run.json ---------------------------------------------------------------------


def _req(doc: Mapping, key: str, where: str = ""):
    if key not in doc:
        raise EvalError(f"run.json: {where}{key} is required")
    return doc[key]


def _check_number(value, where: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or (
            isinstance(value, float) and not math.isfinite(value)):
        raise EvalError(f"run.json: {where} is a finite number or null")


def _check_time(value, where: str, required: bool = True) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not _TIME.match(value):
        raise EvalError(f"run.json: {where} is a UTC time like 2026-10-24T21:30:00Z")


def check_run(doc: object, *, suite: Mapping | None = None,
              metrics: Mapping[str, Mapping] | None = None) -> dict:
    """§4: everything `record` and `publish` refuse. With the suite's entry and
    the registered metrics, the views and metric names are checked too."""
    if not isinstance(doc, Mapping):
        raise EvalError("run.json is one JSON object")
    if doc.get("schema") != RUN_SCHEMA:
        raise EvalError(f"run.json: schema is {RUN_SCHEMA!r}")
    s = _req(doc, "suite")
    if not isinstance(s, Mapping):
        raise EvalError("run.json: suite is {name, digest}")
    check_suite(_req(s, "name", "suite."))
    if not isinstance(_req(s, "digest", "suite."), str) or not _HEX64.match(s["digest"]):
        raise EvalError("run.json: suite.digest is 64 hex digits")

    sub = _req(doc, "subject")
    if not isinstance(sub, Mapping):
        raise EvalError("run.json: subject is an object")
    engine = check_part(_req(sub, "engine", "subject."), "engine", MAX_ENGINE)
    if not isinstance(_req(sub, "engine_version", "subject."), str) or not sub[
            "engine_version"].strip():
        raise EvalError("run.json: subject.engine_version is a release name")
    commit = _req(sub, "commit", "subject.")
    if commit is not None and (not isinstance(commit, str) or not _HEX40.match(commit)):
        raise EvalError("run.json: subject.commit is 40 lower-case hex digits, or null")
    if _req(sub, "dirty", "subject.") not in (True, False, None):
        raise EvalError("run.json: subject.dirty is true, false or null")
    _check_time(sub.get("commit_time"), "subject.commit_time", required=False)
    if sub.get("variant") is not None:
        check_part(sub["variant"], "variant", MAX_VARIANT)
    config = _req(sub, "config", "subject.")
    params = config.get("params") if isinstance(config, Mapping) else None
    if not isinstance(params, Mapping):
        raise EvalError("run.json: subject.config.params is an object")
    for k, v in params.items():
        if not isinstance(k, str) or not k:
            raise EvalError("run.json: subject.config.params keys are strings")
        if v is not None and not isinstance(v, str | int | float | bool):
            raise EvalError(f"run.json: subject.config.params.{k} is a string, number, "
                            "boolean or null: flatten it, keys joined with '.'")
        if isinstance(v, float) and not math.isfinite(v):
            raise EvalError(f"run.json: subject.config.params.{k} is not finite")
    if len(canonical(dict(params))) > MAX_PARAMS:
        raise EvalError("run.json: subject.config.params is over 64 KB")
    if _req(sub, "config_digest", "subject.") != config_digest(params):
        raise EvalError("run.json: subject.config_digest is not the digest of "
                        "subject.config.params")

    inputs = _req(doc, "inputs")
    if not isinstance(inputs, list) or not inputs:
        raise EvalError("run.json: inputs is a non-empty list")
    for i in inputs:
        if not isinstance(i, Mapping) or i.get("role") not in ROLES:
            raise EvalError(f"run.json: each input has a role, one of {', '.join(ROLES)}")
        _layout.check_name(_req(i, "name", "inputs[]."))
        v = _req(i, "version", "inputs[].")
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise EvalError("run.json: inputs[].version is the version read, a whole number")
        if i.get("path") is not None:
            _layout.check_path(i["path"])
        if not isinstance(_req(i, "digest", "inputs[]."), str) or not _HEX64.match(i["digest"]):
            raise EvalError("run.json: inputs[].digest is 64 hex digits")
        cls = _req(i, "data_class", "inputs[].")
        if cls == "client":
            raise Refused(CLIENT_REFUSAL, status=403)
        if cls not in ("synthetic", "public"):
            raise EvalError("run.json: inputs[].data_class is synthetic or public")
        if _req(i, "sealed", "inputs[].") not in (True, False):
            raise EvalError("run.json: inputs[].sealed is true or false")
    if sum(1 for i in inputs if i["role"] == "scored") != 1:
        raise EvalError("run.json: exactly one input has role 'scored'")

    m = _req(doc, "metrics")
    if not isinstance(m, Mapping) or "all" not in m:
        raise EvalError("run.json: metrics is keyed by view, and 'all' is required")
    for view, values in m.items():
        check_part(view, "view", _layout.MAX_SEGMENT)
        if suite is not None and view not in suite.get("views", {}):
            raise EvalError(f"run.json: view {view!r} is not declared by suite "
                            f"{s['name']}; declared: {', '.join(suite['views'])}")
        if not isinstance(values, Mapping):
            raise EvalError(f"run.json: metrics.{view} maps metric names to numbers")
        for name, value in values.items():
            check_part(name, "metric", MAX_SUITE)
            if metrics is not None and name not in metrics:
                raise EvalError(f"run.json: metric {name!r} is not registered; registered: "
                                f"{', '.join(sorted(metrics)) or '(none)'}")
            _check_number(value, f"metrics.{view}.{name}")

    sc = _req(doc, "scorer")
    if not isinstance(sc, Mapping):
        raise EvalError("run.json: scorer is {name, version, digest, rescoreable}")
    for key in ("name", "version"):
        if not isinstance(_req(sc, key, "scorer."), str) or not sc[key]:
            raise EvalError(f"run.json: scorer.{key} is a non-empty string")
    if not isinstance(_req(sc, "digest", "scorer."), str) or not _HEX64.match(sc["digest"]):
        raise EvalError("run.json: scorer.digest is 64 hex digits")
    resc = _req(sc, "rescoreable", "scorer.")
    if not isinstance(resc, Mapping) or not all(isinstance(v, bool) for v in resc.values()):
        raise EvalError("run.json: scorer.rescoreable maps metric names to booleans")
    missing = sorted({n for values in m.values() for n in values} - set(resc))
    if missing:
        raise EvalError(f"run.json: scorer.rescoreable does not say {', '.join(missing)}")

    gate = doc.get("gate")
    if gate is not None:
        if not isinstance(gate, Mapping):
            raise EvalError("run.json: gate is an object or null")
        check_run_id(_req(gate, "against", "gate."))
        if not isinstance(_req(gate, "tolerances_sha", "gate."), str) or not _HEX64.match(
                gate["tolerances_sha"]):
            raise EvalError("run.json: gate.tolerances_sha is 64 hex digits")
        if _req(gate, "verdict", "gate.") not in VERDICTS:
            raise EvalError(f"run.json: gate.verdict is one of {', '.join(VERDICTS)}")
        if gate["verdict"] == "incomparable" and not gate.get("reason"):
            raise EvalError("run.json: an incomparable gate says why in gate.reason")
        if not isinstance(gate.get("failed", []), list):
            raise EvalError("run.json: gate.failed is a list")

    reason = _req(doc, "reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_REASON or (
            "\n" in reason):
        raise EvalError(f"run.json: reason is one line, at most {MAX_REASON} characters")
    if doc.get("experiment") is not None:
        check_part(doc["experiment"], "experiment", MAX_SUITE)
    notes = doc.get("notes")
    if notes is not None and (not isinstance(notes, str) or len(notes.encode()) > MAX_NOTES):
        raise EvalError("run.json: notes is markdown, at most 64 KB")
    kind = _req(doc, "kind")
    if kind not in KINDS:
        raise EvalError(f"run.json: kind is one of {', '.join(KINDS)}")
    if kind == "rescore":
        check_run_id(doc.get("rescored_from"))
    elif doc.get("rescored_from") is not None:
        raise EvalError("run.json: rescored_from is only for kind 'rescore'")
    if _req(doc, "status") not in STATUSES:
        raise EvalError(f"run.json: status is one of {', '.join(STATUSES)}")
    if not isinstance(doc.get("progress", []), list):
        raise EvalError("run.json: progress is a list")

    env = _req(doc, "env")
    if not isinstance(env, Mapping) or env.get("class") not in ENV_CLASSES:
        raise EvalError(f"run.json: env.class is one of {', '.join(ENV_CLASSES)}")
    for key in ("cpu_model", "cores", "threads", "memory_bytes", "os", "python"):
        _req(env, key, "env.")
    load = _req(env, "load", "env.")
    if not isinstance(load, Mapping):
        raise EvalError("run.json: env.load is {start, end}")
    for end in ("start", "end"):
        probe = _req(load, end, "env.load.")
        if not isinstance(probe, Mapping):
            raise EvalError(f"run.json: env.load.{end} is {{loadavg_1m, cpu_busy_pct}}")
        for key in ("loadavg_1m", "cpu_busy_pct"):
            _check_number(probe.get(key), f"env.load.{end}.{key}")

    _check_time(_req(doc, "started_at"), "started_at")
    _check_time(_req(doc, "ended_at"), "ended_at")
    check_run_id(_req(doc, "run_id"), engine=engine, commit=commit)
    return dict(doc)


def run_size_ok(raw: bytes) -> None:
    if len(raw) > MAX_RUN_JSON:
        raise EvalError(f"run.json is {len(raw)} bytes; at most {MAX_RUN_JSON}")


def is_scheduled(doc: Mapping) -> bool:
    trig = doc.get("trigger") or {}
    return bool(doc.get("kind") == "ci" and trig.get("event") == "schedule"
                and trig.get("default_branch") is True)


def strip_sealed(doc: Mapping, fields: Iterable[str]) -> tuple[dict, list[str]]:
    """Every key, at any depth, whose name is in `fields`; and the key paths
    removed, sorted. `publication` is the library's own and is never walked."""
    drop = set(fields)
    clash = sorted(drop & {n for v in (doc.get("metrics") or {}).values()
                           if isinstance(v, Mapping) for n in v})
    if clash:
        raise EvalError(f"sealed field {', '.join(clash)} is also a metric name: stripping it "
                        "would drop the score. Rename the metric")
    removed: list[str] = []

    def walk(value, path: str):
        if isinstance(value, Mapping):
            out = {}
            for k, v in value.items():
                here = f"{path}.{k}" if path else k
                if k in drop:
                    removed.append(here)
                    continue
                out[k] = v if here == "publication" else walk(v, here)
            return out
        if isinstance(value, list):
            return [walk(v, f"{path}[{n}]") for n, v in enumerate(value)]
        return value

    return walk(doc, ""), sorted(removed)


def run_metadata(doc: Mapping, suite: Mapping) -> dict:
    """§6: at most 1,024 bytes, and nothing else."""
    inputs = doc["inputs"]
    scored = next(i for i in inputs if i["role"] == "scored")
    if any(i["data_class"] == "client" for i in inputs):
        raise Refused(CLIENT_REFUSAL, status=403)
    public = [i for i in inputs if i["data_class"] == "public"]
    sealed = any(i["sealed"] for i in inputs)
    gate = doc.get("gate")
    why = gate.get("reason") if gate else None
    # The gate's own reason, one line: what the regression email says (§12.3).
    gate_reason = " ".join(why.split())[:120] if isinstance(why, str) and why.strip() else None
    sub = doc["subject"]
    headline = {}
    for name in [suite["primary"], *suite.get("tie_break", [])]:
        value = doc["metrics"]["all"].get(name)
        if value is not None and name not in headline and len(headline) < HEADLINE:
            headline[name] = sig6(value)
    ev = {"schema": RUN_SCHEMA, "run_id": doc["run_id"], "suite": doc["suite"]["name"],
          "engine": sub["engine"], "engine_version": sub["engine_version"],
          "variant": sub.get("variant"), "kind": doc["kind"], "scheduled": is_scheduled(doc),
          "status": doc["status"], "verdict": gate["verdict"] if gate else None,
          "against": gate["against"] if gate else None, "gate_reason": gate_reason,
          "reason": doc["reason"][:120], "experiment": doc.get("experiment"),
          "headline": headline}
    meta = {"description": f"evaluation run {doc['run_id']}",
            "source": f"{scored['name']}@{scored['version']}",
            "data_class": "public" if public else "synthetic",
            "ci_allowed": not sealed, "eval": ev}
    if public:
        licences = []
        for i in public:
            lic = i.get("licence")
            if not isinstance(lic, str) or not lic.strip():
                raise EvalError(f"input {i['name']} is public and its licence is unknown")
            if lic not in licences:
                licences.append(lic)
        meta["licence"] = "; ".join(licences)
    while len(canonical(meta)) > MAX_RUN_METADATA and ev["headline"]:
        ev["headline"].popitem()
    if len(canonical(meta)) > MAX_RUN_METADATA:
        ev["reason"] = ev["reason"][:40]
        if ev["gate_reason"]:
            ev["gate_reason"] = ev["gate_reason"][:40]
    if len(canonical(meta)) > MAX_RUN_METADATA:
        raise EvalError(f"the run's metadata is over {MAX_RUN_METADATA} bytes even with no "
                        "headline and 40-character reasons; shorten the names")
    return meta


def registry_metadata(kind: str, generation: int, **extra) -> dict:
    meta = {"description": f"evaluation registry: {kind}", "source": "evaluation registry",
            "data_class": "synthetic", "ci_allowed": True,
            "eval": {"registry": kind, "generation": generation, **extra}}
    if len(canonical(meta)) > MAX_REGISTRY_METADATA:
        raise EvalError(f"the {kind} registry metadata would be over 16 KB; the gateway "
                        "refuses it")
    return meta


def check_line_id(value: object) -> str:
    if not isinstance(value, str) or not _LINE_ID.match(value):
        raise EvalError(f"{value!r} is not a registry line id")
    return value


def check_date(value: object, what: str) -> str:
    if not isinstance(value, str) or not _DATE.match(value):
        raise EvalError(f"{what} is a date, YYYY-MM-DD")
    return value
