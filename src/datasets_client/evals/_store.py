"""Publishing runs and writing the registry, over any backend `Datasets` reads.

```
<ns>-evals/runs/<suite>/<run-id>           one dataset per run, never versioned
<ns>-evals/registry/g<N>/{suites,metrics,pins,notes,contamination}
```

**The registry is append-only by construction.** Every write is a read, a new
version whose file keeps the previous one byte for byte as its prefix, and a
push naming the version read as its parent. A gateway lets only the token that
created a registry dataset version it, which is what keeps a CI token from
pinning; a folder has no token, and says so in what it cannot answer.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from datasets_client import _layout
from datasets_client._client import Datasets
from datasets_client._layout import DatasetInfo, Version
from datasets_client.errors import (
    DatasetError,
    DatasetUnavailable,
    Refused,
    Retracted,
)
from datasets_client.evals import _core
from datasets_client.evals._core import EvalError
from datasets_client.evals._record import RUN_FILE, _sealed_declared, read_run

ARTIFACTS = "artifacts"
WRITE_RETRIES = 3
EVALS_VARIABLE = "DATASETS_EVALS"
_GEN = re.compile(r"^g([1-9][0-9]{0,8})$")


def _library_version() -> str:
    from datasets_client import __version__
    return __version__


def _author(given: str | None) -> str:
    if given:
        return given
    try:
        out = subprocess.run(["git", "config", "user.email"], capture_output=True,  # noqa: S603,S607
                             text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _line_id(kind: str) -> str:
    return (f"{kind}-{_core.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}")


def backend_kind(ds: Datasets) -> str:
    name = type(ds.backend).__name__
    if name == "GatewayBackend":
        return "gateway"
    url = (ds.url or "").lower()
    if url.startswith("az://"):
        return "azure"
    if url.startswith("s3://"):
        return "s3"
    return "file"


@dataclass
class Registry:
    """One generation's five documents, as read. `versions` is the version of
    each dataset that was read, for a write's `parent`."""

    generation: int | None
    suites: dict = field(default_factory=lambda: {"schema": _core.SUITES_SCHEMA, "suites": {}})
    metrics: dict = field(default_factory=lambda: {"schema": _core.METRICS_SCHEMA,
                                                   "metrics": []})
    pins: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    contamination: list[dict] = field(default_factory=list)
    raw: dict[str, bytes] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)

    @property
    def metric_table(self) -> dict[str, dict]:
        return _core.metric_table(self.metrics)

    def current_pins(self) -> dict[str, dict[str, str]]:
        """suite -> subject -> run id; the last line naming a pair wins."""
        out: dict[str, dict[str, str]] = {}
        for p in self.pins:
            subject = _core.subject_of(p["engine"], p.get("variant"))
            out.setdefault(p["suite"], {})[subject] = p["run_id"]
        return {s: dict(sorted(v.items())) for s, v in sorted(out.items())}

    def live_notes(self) -> list[dict]:
        """Notes no later note retracts."""
        gone = {n.get("retracts") for n in self.notes if n.get("retracts")}
        return [n for n in self.notes if n["id"] not in gone]

    def invalid(self) -> dict[str, str]:
        """run id -> the note id that marks it invalid."""
        out: dict[str, str] = {}
        for n in self.live_notes():
            if n.get("tag") == "invalid":
                for rid in n.get("runs") or []:
                    out.setdefault(rid, n["id"])
        return out


@dataclass
class Run:
    """One published run: its dataset, its `run.json` and when it registered."""

    name: str
    doc: dict
    registered_at: str | None
    version: Version | None = None

    @property
    def run_id(self) -> str:
        return self.doc["run_id"]

    @property
    def suite(self) -> str:
        return self.doc["suite"]["name"]

    @property
    def engine(self) -> str:
        return self.doc["subject"]["engine"]

    @property
    def variant(self) -> str | None:
        return self.doc["subject"].get("variant")

    @property
    def subject(self) -> str:
        return _core.subject_of(self.engine, self.variant)

    @property
    def scored(self) -> dict:
        return next(i for i in self.doc["inputs"] if i["role"] == "scored")

    @property
    def dataset(self) -> str:
        return self.scored["name"]

    @property
    def sealed(self) -> bool:
        return any(i.get("sealed") for i in self.doc["inputs"])

    @property
    def verdict(self) -> str | None:
        gate = self.doc.get("gate")
        return gate.get("verdict") if gate else None

    def value(self, metric: str, view: str = "all"):
        return (self.doc.get("metrics", {}).get(view) or {}).get(metric)


class Evals:
    """The evaluation namespace `<ns>-evals`, over whatever `Datasets` reads."""

    def __init__(self, evals: str | None = None, *, ds: Datasets | None = None) -> None:
        ns = evals if evals is not None else os.environ.get(EVALS_VARIABLE)
        if not ns:
            raise EvalError(f"name the evaluation namespace: --evals <ns>-evals or "
                            f"{EVALS_VARIABLE}")
        self.ns = _core.check_evals(ns)
        self.ds = ds or Datasets()
        self._registry: Registry | None = None
        self._runs: list[Run] | None = None
        self.skipped: list[dict] = []

    @property
    def backend(self) -> str:
        return backend_kind(self.ds)

    # --- the registry: reading --------------------------------------------------

    def generations(self) -> dict[int, set[str]]:
        """generation -> the registry datasets it holds."""
        out: dict[int, set[str]] = {}
        for info in self.ds.list(f"{self.ns}/registry/"):
            parts = info.name.split("/")
            if len(parts) != 4 or info.latest_version is None:
                continue
            m = _GEN.match(parts[2])
            if m and parts[3] in _core.REGISTRY:
                out.setdefault(int(m.group(1)), set()).add(parts[3])
        return out

    def generation(self) -> int | None:
        """The highest generation under which `pins` exists."""
        found = [g for g, kinds in self.generations().items() if "pins" in kinds]
        return max(found) if found else None

    def _read_file(self, name: str, kind: str) -> tuple[bytes, Version]:
        v = self.ds.resolve(name)
        got = self.ds.fetch(name, _core.REGISTRY_FILES[kind], version=v.version)
        return got.path.read_bytes(), v

    def registry(self, *, fresh: bool = False) -> Registry:
        if self._registry is not None and not fresh:
            return self._registry
        gen = self.generation()
        reg = Registry(generation=gen)
        if gen is not None:
            for kind in _core.REGISTRY:
                raw, v = self._read_file(_core.registry_name(self.ns, gen, kind), kind)
                reg.raw[kind], reg.versions[kind] = raw, v.version
                reg.metadata[kind] = v.metadata
                if kind == "suites":
                    reg.suites = _core.check_suites_doc(_json(raw, "suites.json"))
                elif kind == "metrics":
                    reg.metrics = _core.check_metrics_doc(_json(raw, "metrics.json"))
                else:
                    setattr(reg, kind, _core.parse_jsonl(raw, _core.REGISTRY_FILES[kind]))
        self._registry = reg
        return reg

    def require_registry(self) -> Registry:
        reg = self.registry()
        if reg.generation is None:
            raise EvalError(f"{self.ns} has no registry generation yet: run `evals init` "
                            "with the dev token")
        return reg

    def suite(self, name: str) -> dict:
        suites = self.require_registry().suites["suites"]
        if name not in suites:
            raise EvalError(f"suite {name!r} is not registered; registered: "
                            f"{', '.join(sorted(suites)) or '(none)'}")
        return suites[name]

    # --- the registry: writing --------------------------------------------------

    def _push(self, name: str, files: Mapping[str, bytes], *, parent: int | None,
              metadata: Mapping) -> dict:
        with tempfile.TemporaryDirectory(prefix="evals-") as tmp:
            for rel, body in files.items():
                target = Path(tmp).joinpath(*rel.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            return self.ds.push(tmp, name, parent=parent, metadata=metadata)

    def init(self) -> dict:
        """Generation 1, all five datasets empty, when no generation exists.
        Re-running finishes a half-done init; `pins` goes last."""
        gens = self.generations()
        if any("pins" in kinds for kinds in gens.values()):
            raise EvalError(f"{self.ns} already has generation g{self.generation()}; "
                            "`evals rehome` starts the next one")
        return self._create_generation(1, {k: _empty(k) for k in _core.REGISTRY},
                                       {"suites": {"cadence": {}, "suites": []},
                                        "pins": {"pins": {}}},
                                       have=gens.get(1, set()))

    def _create_generation(self, gen: int, files: Mapping[str, bytes],
                           extras: Mapping[str, Mapping], have: set[str]) -> dict:
        created = []
        for kind in [k for k in _core.REGISTRY if k != "pins"] + ["pins"]:
            name = _core.registry_name(self.ns, gen, kind)
            if kind in have:
                raw, _v = self._read_file(name, kind)
                if raw != files[kind]:
                    raise EvalError(f"{name} already exists with other content; a half-done "
                                    "generation must be finished by the token that began it")
                continue
            self._push(name, {_core.REGISTRY_FILES[kind]: files[kind]}, parent=None,
                       metadata=_core.registry_metadata(kind, gen, **extras.get(kind, {})))
            created.append(name)
        self._registry = None
        return {"generation": gen, "created": created}

    def rehome(self) -> dict:
        """§8.7: generation N+1, a copy of N, created by the token in hand."""
        reg = self.require_registry()
        new = reg.generation + 1
        extras = {k: {key: val for key, val in (reg.metadata[k].get("eval") or {}).items()
                      if key not in ("registry", "generation")} for k in _core.REGISTRY}
        return self._create_generation(new, reg.raw, extras,
                                       have=self.generations().get(new, set()))

    def _append(self, kind: str, line: Mapping, *, check=None) -> dict:
        """Read, append one line, push with the version read as parent; a `409`
        (somebody else's version landed) retries from a fresh read."""
        for attempt in range(WRITE_RETRIES + 1):
            reg = self.registry(fresh=True)
            if reg.generation is None:
                raise EvalError(f"{self.ns} has no registry generation: run `evals init`")
            if check is not None:
                check(reg)
            before = reg.raw[kind]
            if before and not before.endswith(b"\n"):
                before += b"\n"
            after = before + _core.jsonl_line(line)
            if not after.startswith(reg.raw[kind]):
                raise EvalError(f"{kind}: a new version must keep every earlier line")
            extra = {}
            if kind == "pins":
                reg.pins.append(dict(line))
                extra = {"pins": reg.current_pins()}
            name = _core.registry_name(self.ns, reg.generation, kind)
            meta = _core.registry_metadata(kind, reg.generation, **extra)
            try:
                got = self._push(name, {_core.REGISTRY_FILES[kind]: after},
                                 parent=reg.versions[kind], metadata=meta)
            except Refused as exc:
                if exc.status == 409 and attempt < WRITE_RETRIES:
                    continue
                raise
            self._registry = None
            return {"name": got["name"], "version": got["version"], "line": dict(line)}
        raise EvalError(f"{kind}: kept losing to newer versions")  # pragma: no cover

    def _replace(self, kind: str, body: bytes, doc: Mapping) -> dict:
        for attempt in range(WRITE_RETRIES + 1):
            reg = self.registry(fresh=True)
            if reg.generation is None:
                raise EvalError(f"{self.ns} has no registry generation: run `evals init`")
            if kind == "suites":
                _core.check_suites_doc(doc, reg.metric_table)
                # The names too: the alert sweep opens nothing for a suite the
                # registry does not name (§12.1).
                extra = {"cadence": _core.cadence_map(doc), "suites": sorted(doc["suites"])}
            else:
                extra = {}
            if reg.raw[kind] == body:
                return {"name": _core.registry_name(self.ns, reg.generation, kind),
                        "version": reg.versions[kind], "changed": False}
            name = _core.registry_name(self.ns, reg.generation, kind)
            try:
                got = self._push(name, {_core.REGISTRY_FILES[kind]: body},
                                 parent=reg.versions[kind],
                                 metadata=_core.registry_metadata(kind, reg.generation,
                                                                  **extra))
            except Refused as exc:
                if exc.status == 409 and attempt < WRITE_RETRIES:
                    continue
                raise
            self._registry = None
            return {"name": got["name"], "version": got["version"], "changed": True}
        raise EvalError(f"{kind}: kept losing to newer versions")  # pragma: no cover

    def register_suites(self, path: str | os.PathLike) -> dict:
        body = Path(path).read_bytes()
        return self._replace("suites", body, _json(body, str(path)))

    def register_metrics(self, path: str | os.PathLike) -> dict:
        body = Path(path).read_bytes()
        doc = _core.check_metrics_doc(_json(body, str(path)))
        return self._replace("metrics", body, doc)

    def pin(self, run_id: str, *, reason: str | None = None, author: str | None = None,
            suite: str | None = None) -> dict:
        """§8.4: refuses an unpinnable run, naming the rule."""
        run = self.find_run(run_id, suite=suite)
        line = {"id": _line_id("p"), "at": _core.iso(_core.now()), "author": _author(author),
                "suite": run.suite, "engine": run.engine, "variant": run.variant,
                "run_id": run.run_id, "reason": reason}

        def check(reg: Registry) -> None:
            why = unpinnable(run, reg)
            if why:
                raise EvalError(f"{run.run_id} is not pinnable: {why}")

        return self._append("pins", line, check=check)

    def note(self, text: str, *, runs: Sequence[str] = (), experiment: str | None = None,
             tag: str | None = None, retracts: str | None = None,
             author: str | None = None) -> dict:
        if not runs and not experiment:
            raise EvalError("a note names --run or --experiment, at least one")
        if tag is not None and tag not in _core.NOTE_TAGS:
            raise EvalError(f"tag is one of {', '.join(_core.NOTE_TAGS)}")
        if not isinstance(text, str) or not text.strip() or len(
                text.encode()) > _core.MAX_NOTE_TEXT:
            raise EvalError("a note's text is markdown, at most 8 KB")
        for rid in runs:
            _core.check_run_id(rid)
        if experiment is not None:
            _core.check_part(experiment, "experiment", _core.MAX_SUITE)
        line = {"id": _line_id("n"), "at": _core.iso(_core.now()), "author": _author(author),
                "runs": list(runs), "experiment": experiment, "tag": tag, "text": text,
                "retracts": retracts}

        def check(reg: Registry) -> None:
            if retracts is not None:
                _core.check_line_id(retracts)
                if retracts not in {n["id"] for n in reg.notes}:
                    raise EvalError(f"note {retracts} is not in the notes registry")

        return self._append("notes", line, check=check)

    def contaminate(self, *, kind: str, engine: str, dataset: str, since: str,
                    reason: str, view: str | None = None, ref: str | None = None,
                    author: str | None = None) -> dict:
        if kind not in _core.CONTAMINATION_KINDS:
            raise EvalError(f"kind is one of {', '.join(_core.CONTAMINATION_KINDS)}")
        _core.check_part(engine, "engine", _core.MAX_ENGINE)
        name, version = _layout.parse_ref(dataset)
        if version is not None:
            raise EvalError("a contamination record names a dataset without a version")
        if view is not None:
            _core.check_part(view, "view", _layout.MAX_SEGMENT)
        _core.check_date(since, "since")
        if not reason or not reason.strip():
            raise EvalError("a contamination record says why")
        line = {"id": _line_id("c"), "at": _core.iso(_core.now()), "author": _author(author),
                "kind": kind, "engine": engine, "dataset": name, "view": view,
                "since": since, "reason": reason, "ref": ref}
        return self._append("contamination", line)

    # --- runs: reading ------------------------------------------------------------

    def run_infos(self) -> list[DatasetInfo]:
        return [i for i in self.ds.list(f"{self.ns}/runs/")
                if len(i.name.split("/")) == 4 and i.latest_version is not None]

    def _load(self, info: DatasetInfo) -> Run | None:
        try:
            got = self.ds.fetch(info.name, RUN_FILE)
        except Retracted:
            self.skipped.append({"run_id": info.name.rsplit("/", 1)[1], "why": "retracted"})
            return None
        raw = got.path.read_bytes()
        try:
            doc = json.loads(raw.decode("utf-8"))
            _core.check_run(doc)
        except (ValueError, DatasetError) as exc:
            msg = exc.message if isinstance(exc, DatasetError) else "not JSON"
            self.skipped.append({"run_id": info.name.rsplit("/", 1)[1],
                                 "why": f"unreadable run.json: {msg}"})
            return None
        when = info.published_at or (doc.get("publication") or {}).get("at")
        return Run(name=info.name, doc=doc, registered_at=_normal_time(when))

    def runs(self) -> list[Run]:
        """Every visible run, oldest registration first."""
        if self._runs is None:
            self.skipped = []
            loaded = [r for r in map(self._load, self.run_infos()) if r is not None]
            self._runs = sorted(loaded, key=lambda r: (r.registered_at or "", r.run_id))
        return self._runs

    def find_run(self, run_id: str, *, suite: str | None = None) -> Run:
        _core.check_run_id(run_id)
        if suite is not None:
            info_name = _core.run_name(self.ns, _core.check_suite(suite), run_id)
            infos = [i for i in self.run_infos() if i.name == info_name]
        else:
            infos = [i for i in self.run_infos() if i.name.rsplit("/", 1)[1] == run_id]
        if not infos:
            raise EvalError(f"run {run_id} is not published in {self.ns}", status=404)
        if len(infos) > 1:
            raise EvalError(f"run {run_id} is in several suites; pass --suite")
        run = self._load(infos[0])
        if run is None:
            raise EvalError(f"run {run_id} cannot be read: {self.skipped[-1]['why']}",
                            status=404)
        return run

    def writer(self, run: Run) -> dict | None:
        v = self.ds.resolve(run.name)
        return v.writer

    # --- publishing ---------------------------------------------------------------

    def publish(self, folder: str | os.PathLike) -> dict:
        """§9.2: validate, re-check digests where readable, strip a sealed run,
        push as version 1. A retry of the same run is success."""
        folder = Path(folder)
        reg = self.require_registry()
        doc = read_run(folder)
        suite_name = (doc.get("suite") or {}).get("name") if isinstance(doc, Mapping) else None
        suite = self.suite(_core.check_suite(suite_name))
        _core.check_run(doc, suite=suite, metrics=reg.metric_table)
        sealed = any(i["sealed"] for i in doc["inputs"])
        _sealed_declared(suite, suite_name, doc["inputs"])
        if sealed and suite.get("sealed_fields") is None:
            raise EvalError(f"suite {suite_name} reads a sealed input and declares no "
                            "sealed_fields: an empty list is a declaration, a missing one "
                            "is not")
        extra = sorted(p.name for p in folder.iterdir()
                       if p.name not in (RUN_FILE, ARTIFACTS) and not p.name.startswith("."))
        if extra:
            raise EvalError(f"{folder}: a run folder holds run.json and artifacts/ only; "
                            f"also found {', '.join(extra)}")

        checked = self._check_digests(doc)
        published, stripped = (_core.strip_sealed(doc, suite.get("sealed_fields") or [])
                               if sealed else (dict(doc), []))
        # `at` and `library` are kept from an earlier attempt, so a retry pushes
        # the same bytes; what was checked and stripped is recomputed every
        # time, so a block from anywhere else never vouches for this attempt.
        prior = doc.get("publication") if isinstance(doc.get("publication"), Mapping) else {}
        pub = {"at": prior.get("at") if isinstance(prior.get("at"), str)
               else _core.iso(_core.now()),
               "library": prior.get("library") if isinstance(prior.get("library"), str)
               else _library_version(),
               "digest_checked": checked, "stripped": stripped}
        if doc.get("publication") != pub:
            doc["publication"] = pub
            _write_json(folder / RUN_FILE, doc)
        published["publication"] = dict(pub)
        meta = _core.run_metadata(doc, suite)
        name = _core.run_name(self.ns, suite_name, doc["run_id"])

        with tempfile.TemporaryDirectory(prefix="evals-run-") as tmp:
            stage = Path(tmp)
            _write_json(stage / RUN_FILE, published)
            art = folder / ARTIFACTS
            if art.is_dir() and not sealed:
                shutil.copytree(art, stage / ARTIFACTS)
            listing = _layout.scan(stage)
            try:
                got = self.ds.push(stage, name, parent=None, metadata=meta)
                result = {"name": got["name"], "version": got["version"],
                          "already_published": False}
            except Refused as exc:
                if exc.status != 409:
                    raise
                have = self.ds.resolve(name)
                if [(f.path, f.sha256) for f in have.files] != [
                        (f.path, f.sha256) for f in listing]:
                    raise EvalError(f"run id {doc['run_id']} is already published with "
                                    "different content (a retry with another token can "
                                    "differ in publication.digest_checked)",
                                    status=409) from None
                result = {"name": have.name, "version": have.version,
                          "already_published": True}
        result |= {"run_id": doc["run_id"], "digest_checked": pub["digest_checked"],
                   "stripped": list(pub.get("stripped") or []), "sealed": sealed}
        self._runs = None
        if sealed:
            # A retry that found the run already there recorded it the first time.
            result["spent_holdout"] = ("already published; not recorded again"
                                       if result["already_published"]
                                       else self._record_spent_holdout(doc))
        return result

    def _check_digests(self, doc: Mapping) -> str:
        """`publish` when every input was readable and matched; `record` when any
        was not readable by this credential. A mismatch refuses."""
        every = True
        for i in doc["inputs"]:
            try:
                v = self.ds.resolve(i["name"], version=i["version"])
            except (DatasetUnavailable, Retracted):
                every = False
                continue
            except Refused as exc:
                if exc.status in (400, 403):
                    every = False
                    continue
                raise
            if _layout.data_class_of(v.metadata) == "client":
                raise Refused(f"{i['name']}: {_core.CLIENT_REFUSAL}", status=403)
            if (v.metadata or {}).get("purpose") == "sealed" and not i["sealed"]:
                # The run would publish its artefacts and answer key unstripped.
                raise EvalError(f"{i['name']}@{i['version']} is a sealed input and the run "
                                "says it is not; re-record it", status=409)
            if _core.content_digest(v.files, i.get("path")) != i["digest"]:
                raise EvalError(f"{i['name']}@{i['version']}: its digest is not the one "
                                "recorded; the input changed or the record is wrong",
                                status=409)
        return "publish" if every else "record"

    def _record_spent_holdout(self, doc: Mapping) -> str:
        """§8.6: when this token can write the registry. Says what happened.

        Only runs that STARTED before this one count, and a dataset that already
        has a `spent_holdout` record for this engine since on or before this
        run's start gets no second one: the registry is append-only, so a
        duplicate could never be cleaned up."""
        sub = doc["subject"]
        sealed = [i["name"] for i in doc["inputs"] if i["sealed"]]
        earlier = [r for r in self.runs() if r.run_id != doc["run_id"]
                   and r.doc["started_at"] < doc["started_at"]
                   and r.engine == sub["engine"]
                   and any(i["name"] in sealed and i.get("sealed") for i in r.doc["inputs"])]
        if not any(r.doc["subject"]["engine_version"] != sub["engine_version"]
                   or r.doc["subject"]["config_digest"] != sub["config_digest"]
                   for r in earlier):
            return "not spent"
        day = doc["started_at"][:10]
        try:
            have = self.registry(fresh=True).contamination
            todo = [d for d in sealed if not any(
                c.get("kind") == "spent_holdout" and c.get("engine") == sub["engine"]
                and c.get("dataset") == d and str(c.get("since", "")) <= day for c in have)]
            for dataset in todo:
                self.contaminate(kind="spent_holdout", engine=sub["engine"], dataset=dataset,
                                 since=day, ref=doc["run_id"],
                                 reason="engine changed since the last look")
        except (Refused, DatasetUnavailable) as exc:
            return f"spent; not recorded with this credential ({exc.message})"
        return "spent; recorded" if todo else "spent; already recorded"


def unpinnable(run: Run, reg: Registry) -> str | None:
    doc = run.doc
    if doc["status"] != "complete":
        return f"status is {doc['status']}, not complete"
    if doc["subject"].get("dirty") is not False:
        return "subject.dirty is not false"
    if not doc["subject"].get("commit"):
        return "the commit is unknown"
    if doc["kind"] == "pr":
        return "a pr run is never a baseline"
    if run.sealed:
        return "it read a sealed input"
    if doc.get("input_inferred"):
        return "its inputs were inferred; verify the digests first (§9.5)"
    note = reg.invalid().get(run.run_id)
    if note:
        return f"note {note} marks it invalid"
    return None


def _empty(kind: str) -> bytes:
    if kind == "suites":
        return _core.canonical({"schema": _core.SUITES_SCHEMA, "suites": {}}) + b"\n"
    if kind == "metrics":
        return _core.canonical({"schema": _core.METRICS_SCHEMA, "metrics": []}) + b"\n"
    return b""


def _json(raw: bytes, what: str):
    try:
        return json.loads(raw.decode("utf-8"))
    except ValueError:
        raise EvalError(f"{what} is not JSON") from None


def _write_json(path: Path, doc) -> None:
    raw = (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    _core.run_size_ok(raw)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _normal_time(value) -> str | None:
    when = _core.parse_time(value)
    return _core.iso(when) if when else None
