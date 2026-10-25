"""The query verbs (§10). Each returns one `eval-query/1` object.

**Everything is compared; differences are annotated.** Two runs with another
scorer, another input or another machine are still put side by side, and the
row says how they differ and how much that matters. A reader is never handed a
bare number whose footing changed.

**"Could not answer" is never an empty answer.** What a backend cannot say is
listed under `unavailable`; an unknown metric, suite, engine or dataset is an
error naming the valid ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import timedelta

from datasets_client.backends.http_gateway import TOKEN_VARIABLE
from datasets_client.errors import DatasetError
from datasets_client.evals import _core
from datasets_client.evals._core import EvalError, sig6
from datasets_client.evals._store import ARTIFACTS, Evals, Registry, Run

DEFAULT_LIMIT = 50
MAX_LIMIT = 1000
MAX_EXCLUDED = 20
INCLUDABLE = ("dirty", "pr", "partial", "failed", "invalid", "superseded")
SELECTS = ("pinned", "latest-release", "latest", "best-run")
UNAVAILABLE = "registry not available"
SLURM = {"what": "slurm_overlap", "reason": UNAVAILABLE,
         "detail": "no read path to compute jobs in v1, on any backend"}


# --- the envelope -----------------------------------------------------------------


def envelope(ev: Evals, verb: str, args: Mapping, rows: list, *, limit: int | None = None,
             fields: Sequence[str] | None = None, notes: Iterable[str] = (),
             excluded: Sequence[Mapping] = (), unavailable: Sequence[Mapping] = (),
             extra: Mapping | None = None) -> dict:
    limit = check_limit(limit)
    total = len(rows)
    shown = rows[:limit]
    if fields:
        shown = [{k: r.get(k) for k in fields} if isinstance(r, Mapping) else r
                 for r in shown]
    seen, uniq = set(), []
    for item in unavailable:
        key = (item.get("what"), item.get("reason"))
        if key not in seen:
            seen.add(key)
            uniq.append(dict(item))
    out = {"schema": _core.QUERY_SCHEMA, "verb": verb, "evals": ev.ns, "backend": ev.backend,
           "as_of": _core.iso(_core.now()),
           "args": {k: v for k, v in args.items() if v is not None},
           "total": total, "truncated": total > limit, "rows": shown,
           "notes": list(dict.fromkeys(notes)),
           "excluded": [dict(e) for e in excluded[:MAX_EXCLUDED]],
           "excluded_total": len(excluded), "unavailable": uniq}
    if extra:
        out |= dict(extra)
    return out


def check_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
        raise EvalError(f"--limit is 1 to {MAX_LIMIT}")
    return limit


def _names(values: Iterable[str]) -> str:
    return ", ".join(sorted(set(values))) or "(none)"


def _newest(runs: Iterable[Run]) -> Run | None:
    runs = list(runs)
    return max(runs, key=lambda r: (r.registered_at or "", r.run_id)) if runs else None


# --- one query's context ----------------------------------------------------------


class Query:
    """The registry, the runs and the bookkeeping one answer needs."""

    def __init__(self, ev: Evals, *, include: Iterable[str] = (), view: str = "all") -> None:
        self.ev = ev
        self.reg: Registry = ev.registry()
        self.all = ev.runs()
        self.include = set(include)
        unknown = self.include - set(INCLUDABLE)
        if unknown:
            raise EvalError(f"--include takes {', '.join(INCLUDABLE)}; not {_names(unknown)}")
        self.view = _core.check_part(view, "view", 64)
        self.excluded: list[dict] = list(ev.skipped)
        self._excluded_ids = {e["run_id"] for e in self.excluded}
        self.unavailable: list[dict] = []
        self.notes: list[str] = []
        if self.reg.generation is None:
            self.notes.append(f"{ev.ns} has no registry generation: no suites, metrics, "
                              "pins or notes to read (evals init)")
        self.invalid = self.reg.invalid()
        # Only a rescore that is itself in supersedes its original: a dirty,
        # partial or invalid rescore would otherwise take both runs out.
        self.superseded = {r.doc["rescored_from"]: r.run_id for r in self.all
                           if r.doc["kind"] == "rescore" and r.doc.get("rescored_from")
                           and self._flaw(r) is None}
        self.metrics = self.reg.metric_table
        self.suites = self.reg.suites.get("suites", {})

    # --- vocabulary ---------------------------------------------------------

    def metric_names(self) -> set[str]:
        if self.metrics:
            return set(self.metrics)
        return {m for r in self.all for values in r.doc["metrics"].values() for m in values}

    def check_metric(self, metric: str) -> str:
        valid = self.metric_names()
        if metric not in valid:
            raise EvalError(f"unknown metric {metric!r}; valid: {_names(valid)}")
        return metric

    def metric_for(self, metric: str | None, suites: Iterable[str]) -> str:
        if metric:
            return self.check_metric(metric)
        primaries = {self.suites[s]["primary"] for s in suites if s in self.suites}
        if len(primaries) == 1:
            return self.check_metric(primaries.pop())
        if not primaries:
            raise EvalError("no registered suite gives a primary metric here; pass --metric "
                            f"(valid: {_names(self.metric_names())})")
        raise EvalError(f"the suites here have different primary metrics ({_names(primaries)}); "
                        "pass --metric")

    def primary(self, run: Run) -> str | None:
        """The run's suite's primary metric, else the first it reported."""
        if run.suite in self.suites:
            return self.suites[run.suite]["primary"]
        return next(iter(sorted(run.doc["metrics"]["all"])), None)

    def metric_kind(self, metric: str) -> str | None:
        return (self.metrics.get(metric) or {}).get("kind")

    def better(self, metric: str) -> str | None:
        return (self.metrics.get(metric) or {}).get("better")

    def check_direction(self, metric: str) -> None:
        if metric in self.metrics and self.better(metric) is None:
            raise EvalError(f"{metric} has no direction (better: null): shown and compared, "
                            "never ranked")

    def check_suites(self, names: Iterable[str]) -> list[str]:
        known = set(self.suites) | {r.suite for r in self.all}
        names = list(names)
        bad = [s for s in names if s not in known]
        if bad:
            raise EvalError(f"unknown suite {_names(bad)}; known: {_names(known)}")
        return names

    def datasets(self) -> set[str]:
        return ({_core.scored_input(s) for s in self.suites.values()}
                | {r.dataset for r in self.all})

    def check_dataset(self, name: str) -> str:
        if name not in self.datasets():
            raise EvalError(f"unknown dataset {name!r}; known: {_names(self.datasets())}")
        return name

    def views(self, suites: Iterable[str]) -> set[str]:
        """The views these suites declare, and those their runs reported."""
        suites = set(suites)
        out = {"all"}
        for s in suites:
            if s in self.suites:
                out |= set(self.suites[s]["views"])
        return out | {v for r in self.all if r.suite in suites for v in r.doc["metrics"]}

    def check_view(self, suites: Iterable[str]) -> str:
        """`--view` names a view of these suites: an unknown one would make every
        value null, which reads as missing data rather than a typo."""
        known = self.views(suites)
        if self.view not in known:
            raise EvalError(f"unknown view {self.view!r}; declared: {_names(known)}")
        return self.view

    def check_engine(self, engine: str) -> str:
        known = {r.engine for r in self.all}
        if engine not in known:
            raise EvalError(f"no run of engine {engine!r}; engines: {_names(known)}")
        return engine

    # --- what is in, what is out ----------------------------------------------

    def _flaw(self, run: Run) -> tuple[str, str] | None:
        """Why a run is out, other than being superseded."""
        doc = run.doc
        if doc["subject"].get("dirty") is not False:
            return "dirty", "dirty"
        if doc["kind"] == "pr":
            return "pr", "pr"
        if doc["status"] == "partial":
            return "partial", "partial"
        if doc["status"] == "failed":
            return "failed", "failed"
        note = self.invalid.get(run.run_id)
        if note:
            return "invalid", f"invalid (note {note})"
        return None

    def why_out(self, run: Run) -> tuple[str, str] | None:
        flaw = self._flaw(run)
        if flaw is not None:
            return flaw
        by = self.superseded.get(run.run_id)
        if by:
            return "superseded", f"superseded (rescored by {by})"
        return None

    def included(self, runs: Iterable[Run]) -> list[Run]:
        out = []
        for r in runs:
            why = self.why_out(r)
            if why is None or why[0] in self.include:
                out.append(r)
            elif r.run_id not in self._excluded_ids:
                self._excluded_ids.add(r.run_id)
                self.excluded.append({"run_id": r.run_id, "why": why[1]})
        return out

    def select(self, runs: Sequence[Run], how: str, metric: str | None,
               view: str) -> tuple[Run | None, str | None, int]:
        """§10.1: one representative of these runs (one subject, one dataset)."""
        if how not in SELECTS:
            raise EvalError(f"--select is one of {', '.join(SELECTS)}")
        if how == "best-run" and any(r.sealed for r in runs):
            raise EvalError("--select best-run is refused on a sealed input: picking the "
                            "best of many looks at a hold-out is how it is overfitted")
        inc = self.included(runs)
        if how == "best-run":
            valued = [r for r in inc if r.value(metric, view) is not None]
            ranked = self.order(valued, metric, view, [])
            return (ranked[0] if ranked else None), "best-run", len(inc)
        if how == "pinned":
            pins = self.reg.current_pins()
            for r in inc:
                if pins.get(r.suite, {}).get(r.subject) == r.run_id:
                    return r, "pinned", len(inc)
            how = "latest-release"
        if how == "latest-release":
            return _newest(r for r in inc if r.doc["kind"] == "release"), "latest-release", len(
                inc)
        return _newest(inc), "latest", len(inc)

    # --- ordering ---------------------------------------------------------------

    def _key(self, run: Run, metric: str, view: str):
        v = run.value(metric, view)
        if v is None:
            return (1, 0.0)
        v = sig6(v)
        return (0, -v if self.better(metric) != "down" else v)

    def order(self, runs: Sequence[Run], metric: str, view: str,
              tie_break: Sequence[str]) -> list[Run]:
        return sorted(runs, key=lambda r: (self._key(r, metric, view),
                                           *[self._key(r, m, view) for m in tie_break],
                                           r.subject, r.run_id))

    def ties(self, runs: Sequence[Run], metric: str, view: str) -> list[list[str]]:
        groups: dict[float, list[str]] = {}
        for r in runs:
            v = r.value(metric, view)
            if v is not None:
                groups.setdefault(sig6(v), []).append(r.subject)
        return [sorted(g) for g in groups.values() if len(g) > 1]

    # --- notes a row carries ---------------------------------------------------

    def contamination(self, run: Run, view: str) -> list[str]:
        names = {i["name"] for i in run.doc["inputs"]}
        commit_day = (run.doc["subject"].get("commit_time") or "")[:10] or None
        out = []
        for c in self.reg.contamination:
            if c.get("engine") != run.engine or c.get("dataset") not in names:
                continue
            if commit_day is not None and commit_day < c.get("since", ""):
                continue
            # A record about one view travels on every view, `all` included,
            # which contains it: views are not known to be disjoint (§8.6).
            where = f" on view {c['view']}" if c.get("view") else ""
            out.append(f"contamination: {c['kind']}{where} since {c['since']} "
                       f"({c.get('ref') or c.get('reason')})")
        return out

    def spent_holdout(self, run: Run) -> bool:
        if not run.sealed:
            return False
        sealed = {i["name"] for i in run.doc["inputs"] if i.get("sealed")}
        sub = run.doc["subject"]
        for r in self.all:
            if (r.registered_at or "", r.run_id) >= (run.registered_at or "", run.run_id):
                continue
            if r.engine != run.engine or not any(
                    i["name"] in sealed and i.get("sealed") for i in r.doc["inputs"]):
                continue
            if (r.doc["subject"]["engine_version"] != sub["engine_version"]
                    or r.doc["subject"]["config_digest"] != sub["config_digest"]):
                return True
        return False

    def carried(self, run: Run, view: str) -> list[str]:
        notes = self.contamination(run, view)
        self.notes.extend(notes)
        if run.doc.get("input_inferred"):
            notes.append("inputs inferred")
        if run.doc["status"] != "complete":
            notes.append(f"{run.doc['status']} run")
        if self.spent_holdout(run):
            notes.append("spent_holdout: engine changed since the last look")
        if run.run_id in self.invalid:
            notes.append(f"invalid (note {self.invalid[run.run_id]})")
        return notes

    # --- a row --------------------------------------------------------------------

    def row(self, run: Run, metric: str, view: str, *, selected_by: str | None = None,
            considered: int | None = None) -> dict:
        value = run.value(metric, view)
        out = {"run_id": run.run_id, "suite": run.suite, "dataset": run.dataset,
               "subject": run.subject, "engine_version": run.doc["subject"]["engine_version"],
               "kind": run.doc["kind"], "registered_at": run.registered_at, "metric": metric,
               "value": value, "verdict": run.verdict, "selected_by": selected_by,
               "runs_considered": considered, "comparable_to_leader": None,
               "notes": self.carried(run, view)}
        meta = self.metrics.get(metric) or {}
        if meta.get("kind") == "time":
            out["timing_reference"] = timing_reference(run)
        if meta.get("kind") == "count" and value is not None:
            if meta.get("denominator") is None:
                out["shown"] = f"{value}"
            else:
                total = run.value(meta["denominator"], view)
                out["shown"] = f"{value}/{total if total is not None else '?'}"
        return out

    def timing(self, metric: str) -> None:
        if self.metric_kind(metric) == "time":
            self.unavailable.append(SLURM)

    def done(self, verb: str, args: Mapping, rows: list, *, limit=None, fields=None,
             extra=None) -> dict:
        return envelope(self.ev, verb, args, rows, limit=limit, fields=fields,
                        notes=self.notes, excluded=self.excluded,
                        unavailable=self.unavailable, extra=extra)


# --- comparison -------------------------------------------------------------------


def _loaded(run: Run) -> bool:
    env = run.doc.get("env") or {}
    load = ((env.get("load") or {}).get("start") or {}).get("loadavg_1m")
    threads = env.get("threads")
    return isinstance(load, int | float) and isinstance(threads, int) and load > threads / 2


def timing_reference(run: Run) -> bool:
    """§10.3: a compute-box run, given the machine to itself, and quiet at the start."""
    env = run.doc.get("env") or {}
    load = ((env.get("load") or {}).get("start") or {}).get("loadavg_1m")
    threads = env.get("threads")
    return bool(env.get("class") == "compute-box"
                and (env.get("allocation") or {}).get("exclusive") is True
                and isinstance(load, int | float) and isinstance(threads, int)
                and load < threads / 4)


def differences(a: Run, b: Run, view_a: str = "all", view_b: str = "all") -> list[dict]:
    """§10.3, every difference named, with its severity."""
    out = []
    sa, sb = a.doc["scorer"], b.doc["scorer"]
    for key in ("name", "version", "digest"):
        if sa.get(key) != sb.get(key):
            out.append({"kind": "scorer", "severity": "minor", "field": f"scorer.{key}",
                        "a": sa.get(key), "b": sb.get(key)})
    if a.scored["digest"] != b.scored["digest"] or a.dataset != b.dataset:
        out.append({"kind": "input", "severity": "major", "field": "scored",
                    "a": f"{a.dataset}@{a.scored['version']}",
                    "b": f"{b.dataset}@{b.scored['version']}"})

    def truth(r: Run):
        return sorted((i["name"], i.get("path") or "", i["digest"]) for i in r.doc["inputs"]
                      if i["role"] == "truth")

    if truth(a) != truth(b):
        out.append({"kind": "input", "severity": "major", "field": "truth",
                    "a": [f"{n}{'/' + p if p else ''}" for n, p, _ in truth(a)],
                    "b": [f"{n}{'/' + p if p else ''}" for n, p, _ in truth(b)]})
    if view_a != view_b:
        out.append({"kind": "view", "severity": "major", "field": "view", "a": view_a,
                    "b": view_b})
    if a.doc["suite"]["digest"] != b.doc["suite"]["digest"]:
        out.append({"kind": "view", "severity": "major", "field": "suite.digest",
                    "a": a.doc["suite"]["digest"][:12], "b": b.doc["suite"]["digest"][:12]})
    ea, eb = a.doc["env"], b.doc["env"]
    for key in ("class", "resource", "host", "cpu_model", "allocation"):
        if ea.get(key) != eb.get(key):
            out.append({"kind": "env", "severity": "timing", "field": f"env.{key}",
                        "a": ea.get(key), "b": eb.get(key)})
    if _loaded(a) or _loaded(b):
        out.append({"kind": "env", "severity": "timing", "field": "env.load",
                    "a": ea["load"]["start"].get("loadavg_1m"),
                    "b": eb["load"]["start"].get("loadavg_1m")})
    return out


def comparable(diffs: Sequence[Mapping]) -> str:
    if any(d["severity"] == "major" for d in diffs):
        return "input"
    if any(d["severity"] == "minor" for d in diffs):
        return "scorer"
    return "exact"


def param_changes(a: Mapping, b: Mapping) -> dict:
    """From `a` to `b`, computed at query time and never stored."""
    pa, pb = a["subject"]["config"]["params"], b["subject"]["config"]["params"]
    return {"added": {k: pb[k] for k in sorted(set(pb) - set(pa))},
            "removed": {k: pa[k] for k in sorted(set(pa) - set(pb))},
            "changed": [{"key": k, "from": pa[k], "to": pb[k]}
                        for k in sorted(set(pa) & set(pb)) if pa[k] != pb[k]]}


def _summary(changes: Mapping, cap: int = 20) -> dict:
    return {"added": sorted(changes["added"])[:cap], "removed": sorted(changes["removed"])[:cap],
            "changed": changes["changed"][:cap],
            "counts": {k: len(changes[k]) for k in ("added", "removed", "changed")}}


def _annotate(q: Query, row: dict, run: Run, leader: Run, metric: str, view: str) -> None:
    diffs = differences(leader, run)
    level = comparable(diffs)
    row["comparable_to_leader"] = level
    if level == "scorer":
        row["notes"].append("scorer differs from the leader")
        if run.doc["scorer"]["rescoreable"].get(metric):
            row["notes"].append("rescore to compare exactly")
    elif level == "input":
        row["notes"].append("input differs from the leader: the ground truth may differ")
    if q.metric_kind(metric) == "time" and run is not leader:
        timing = [d["field"] for d in diffs if d["severity"] == "timing"]
        if timing:
            row["notes"].append(f"timing: {', '.join(timing)} differ from the leader")


# --- verbs ------------------------------------------------------------------------


def _scope(q: Query, suites: Sequence[str] | None) -> list[str]:
    if suites:
        return q.check_suites(suites)
    return sorted(set(q.suites) | {r.suite for r in q.all})


def best(ev: Evals, *, dataset: str, metric: str | None = None, select: str = "pinned",
         view: str = "all", include: Iterable[str] = (), limit: int | None = None,
         fields: Sequence[str] | None = None) -> dict:
    q = Query(ev, include=include, view=view)
    q.check_dataset(dataset)
    runs = [r for r in q.all if r.dataset == dataset]
    metric = q.metric_for(metric, {r.suite for r in runs} or {
        s for s, e in q.suites.items() if _core.scored_input(e) == dataset})
    if q.metric_kind(metric) == "context":
        raise EvalError(f"{metric} is a context metric: shown, never ranked")
    q.check_direction(metric)
    q.check_view({r.suite for r in runs} or {
        s for s, e in q.suites.items() if _core.scored_input(e) == dataset})
    q.timing(metric)
    picked, unselected = [], []
    for subject in sorted({r.subject for r in runs}):
        group = [r for r in runs if r.subject == subject]
        run, by, n = q.select(group, select, metric, view)
        if run is not None:
            picked.append((run, by, n))
        else:
            unselected.append(_no_selection(subject, select, n))
    tie = _tie_break(q, [r.suite for r, _, _ in picked])
    ordered = q.order([r for r, _, _ in picked], metric, view, tie)
    meta = {r.run_id: (by, n) for r, by, n in picked}
    leader = next((r for r in ordered if r.value(metric, view) is not None), None)
    rows, major = [], []
    for run in ordered:
        row = q.row(run, metric, view, selected_by=meta[run.run_id][0],
                    considered=meta[run.run_id][1])
        if leader is not None:
            _annotate(q, row, run, leader, metric, view)
        (major if row["comparable_to_leader"] == "input" else rows).append(row)
    rank = {"exact": 0, "scorer": 1, None: 2}
    rows.sort(key=lambda r: rank[r["comparable_to_leader"]])
    args = {"dataset": dataset, "metric": metric, "select": select, "view": view}
    return q.done("best", args, rows, limit=limit, fields=fields,
                  extra={"major": major, "leader": leader.subject if leader else None,
                         "ties": q.ties([r for r, _, _ in picked], metric, view),
                         "no_selection": unselected})


def _no_selection(subject: str, select: str, considered: int) -> dict:
    """A subject with runs and none `--select` can choose (`scoreboard`'s
    `no_selection` cell, for `best`)."""
    return {"subject": subject, "runs_considered": considered,
            "why": "no included run matches --select " + select
                   + (" (no pin, no release); try --select latest" if select == "pinned"
                      else "")}


def _tie_break(q: Query, suites: Iterable[str]) -> list[str]:
    for s in suites:
        if s in q.suites:
            return list(q.suites[s].get("tie_break") or [])
    return []


def scoreboard(ev: Evals, *, metric: str | None = None, datasets: Sequence[str] | None = None,
               suites: Sequence[str] | None = None, select: str = "pinned", view: str = "all",
               include: Iterable[str] = (), limit: int | None = None,
               fields: Sequence[str] | None = None) -> dict:
    q = Query(ev, include=include, view=view)
    scope = _scope(q, suites)
    metric = q.metric_for(metric, scope)
    kind = q.metric_kind(metric)
    if kind in ("count", "context"):
        raise EvalError(f"{metric} is a {kind} metric: only rate and time metrics rank across "
                        "datasets. Use `best --dataset` for one dataset")
    q.check_direction(metric)
    q.check_view(scope)
    q.timing(metric)
    in_scope = [r for r in q.all if r.suite in scope]
    names = sorted({_core.scored_input(q.suites[s]) for s in scope if s in q.suites}
                   | {r.dataset for r in in_scope})
    if datasets:
        bad = [d for d in datasets if d not in names]
        if bad:
            raise EvalError(f"unknown dataset {_names(bad)} for these suites; known: "
                            f"{_names(names)}")
        names = [d for d in names if d in datasets]
    subjects = sorted({r.subject for r in in_scope})
    cells, chosen = [], {}
    for subject in subjects:
        for dataset in names:
            group = [r for r in in_scope if r.subject == subject and r.dataset == dataset]
            if not group:
                cells.append({"subject": subject, "dataset": dataset, "state": "not_run"})
                continue
            run, by, n = q.select(group, select, metric, view)
            if run is None:
                cells.append({"subject": subject, "dataset": dataset, "state": "no_selection"}
                             | _no_selection(subject, select, n))
                continue
            row = q.row(run, metric, view, selected_by=by, considered=n)
            row["state"] = "ok" if row["value"] is not None else "no_value"
            cells.append(row)
            if row["value"] is not None:
                chosen[(subject, dataset)] = run
    coverage = {s: f"{sum(1 for (x, _d) in chosen if x == s)}/{len(names)}" for s in subjects}

    # The common subset: add subjects most-covered first while one remains.
    have = {s: {d for (x, d) in chosen if x == s} for s in subjects}
    ranked, common = [], None
    for s in sorted(subjects, key=lambda s: (-len(have[s]), s)):
        if not have[s]:
            continue
        nxt = have[s] if common is None else common & have[s]
        if nxt:
            ranked.append(s)
            common = nxt
    common = sorted(common or [])
    unranked = [{"subject": s, "why": "no dataset in common with the ranked subjects"
                 if have[s] else "no value on any dataset"} for s in subjects if s not in ranked]

    leads = {s: 0 for s in ranked}
    footing = {s: "exact" for s in ranked}
    worse = {"exact": 0, "scorer": 1, "input": 2}
    tie = _tie_break(q, scope)
    for d in common:
        runs = [chosen[(s, d)] for s in ranked]
        best_run = q.order(runs, metric, view, tie)[0]
        top = sig6(best_run.value(metric, view))
        for r in runs:
            if sig6(r.value(metric, view)) == top:
                leads[r.subject] += 1
        for r in runs:
            row = next(c for c in cells if c.get("run_id") == r.run_id and c["dataset"] == d)
            _annotate(q, row, r, best_run, metric, view)
            level = row["comparable_to_leader"]
            if worse[level] > worse[footing[r.subject]]:
                footing[r.subject] = level

    def mean(s: str, m: str):
        values = [chosen[(s, d)].value(m, view) for d in common]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if len(values) == len(common) and values else None

    down = q.better(metric) == "down"
    order = sorted(ranked, key=lambda s: (-leads[s], (mean(s, metric) or 0) * (1 if down else -1),
                                          s))
    # `comparable`: the worst footing of the subject's cells against each
    # dataset's leader. `input` means a lead or a mean here is over another
    # ground truth on at least one dataset (§10.3), and the ranking says so.
    ranking = [{"rank": n + 1, "subject": s, "leads": leads[s], "of": len(common),
                "mean": mean(s, metric), "coverage": coverage[s], "comparable": footing[s]}
               for n, s in enumerate(order)]
    if any(r["comparable"] == "input" for r in ranking):
        q.notes.append("ranking: some subjects ran on another input than the leader on a "
                       "dataset; see each ranking entry's `comparable`")
    ties: list[list[str]] = []
    for r in ranking:
        same = [x["subject"] for x in ranking if x["leads"] == r["leads"]
                and sig6(x["mean"]) == sig6(r["mean"])]
        if len(same) > 1 and sorted(same) not in ties:
            ties.append(sorted(same))

    tradeoffs = []
    if ranking:
        leader = ranking[0]["subject"]
        times = [m for m, meta in q.metrics.items() if meta.get("kind") == "time"]
        for other in order[1:]:
            for m in times:
                lv, ov = mean(leader, m), mean(other, m)
                if lv is None or ov is None or lv == ov or 0 in (lv, ov):
                    continue
                up = q.better(m) == "up"
                if (ov > lv) if up else (ov < lv):
                    ratio = round((ov / lv) if up else (lv / ov), 2)
                    word = "higher" if up else "faster"
                    tradeoffs.append({"subject": other, "metric": m, "ratio": ratio,
                                      "text": f"{other} {ratio:g}x {word} on {m}"})
    args = {"metric": metric, "datasets": ",".join(datasets) if datasets else None,
            "suites": ",".join(scope), "select": select, "view": view}
    return q.done("scoreboard", args, cells, limit=limit, fields=fields,
                  extra={"cells_total": len(cells), "coverage": coverage,
                         "common_subset": common, "ranking": ranking, "unranked": unranked,
                         "ties": ties, "tradeoffs": tradeoffs})


def _reference(q: Query, run: Run, against: str | None) -> tuple[Run | None, str]:
    """`--against`: `baseline` (the run the gate named, else the pin),
    `previous-in-experiment`, or a run id."""
    how = against or "baseline"
    if how == "baseline":
        gate = run.doc.get("gate")
        rid = gate.get("against") if gate else None
        rid = rid or q.reg.current_pins().get(run.suite, {}).get(run.subject)
        if rid == run.run_id:
            rid = None
    elif how == "previous-in-experiment":
        exp = run.doc.get("experiment")
        earlier = [r for r in q.all if exp and r.doc.get("experiment") == exp
                   and (r.registered_at or "", r.run_id) < (run.registered_at or "", run.run_id)]
        prev = _newest(earlier)
        rid = prev.run_id if prev else None
    else:
        rid = _core.check_run_id(how)
    if rid is None:
        return None, how
    found = next((r for r in q.all if r.run_id == rid), None)
    if found is None:
        q.notes.append(f"--against {how}: run {rid} is not published here")
    return found, how


def history(ev: Evals, *, engine: str, variant: str | None = None,
            dataset: str | None = None, suite: str | None = None, per: str = "version",
            kinds: Sequence[str] | None = None, experiment: str | None = None,
            metric: str | None = None, view: str = "all", against: str | None = None,
            include: Iterable[str] = (), limit: int | None = None,
            fields: Sequence[str] | None = None) -> dict:
    q = Query(ev, include=include, view=view)
    q.check_engine(engine)
    if per not in ("version", "run", "day"):
        raise EvalError("--per is version, run or day")
    if dataset and suite:
        raise EvalError("--dataset or --suite, not both")
    runs = [r for r in q.all if r.engine == engine]
    if variant is not None:
        known = {r.variant for r in runs if r.variant}
        if variant not in known:
            raise EvalError(f"engine {engine} has no run of variant {variant!r}; variants: "
                            f"{_names(known)}")
        runs = [r for r in runs if r.variant == variant]
    if dataset:
        q.check_dataset(dataset)
        runs = [r for r in runs if r.dataset == dataset]
    if suite:
        q.check_suites([suite])
        runs = [r for r in runs if r.suite == suite]
    if kinds:
        bad = [k for k in kinds if k not in _core.KINDS]
        if bad:
            raise EvalError(f"--kind takes {', '.join(_core.KINDS)}; not {_names(bad)}")
        runs = [r for r in runs if r.doc["kind"] in kinds]
    if experiment:
        known = {r.doc["experiment"] for r in q.all if r.engine == engine
                 and r.doc.get("experiment")}
        if experiment not in known:
            raise EvalError(f"engine {engine} has no run in experiment {experiment!r}; "
                            f"experiments: {_names(known)}")
        runs = [r for r in runs if r.doc.get("experiment") == experiment]
    in_scope = {r.suite for r in runs} or ({suite} if suite else set(q.suites))
    metric = q.metric_for(metric, in_scope)
    q.check_view(in_scope)
    q.timing(metric)
    inc = q.included(runs)
    pins = q.reg.current_pins()
    groups: dict[tuple, list[Run]] = {}
    for r in inc:
        if per == "version":
            key = (r.subject, r.dataset, r.doc["subject"]["engine_version"])
        elif per == "day":
            key = (r.subject, r.dataset, (r.registered_at or "")[:10])
        else:
            key = (r.subject, r.dataset, r.run_id)
        groups.setdefault(key, []).append(r)
    rows = []
    for _key, group in groups.items():
        pinned = [r for r in group if pins.get(r.suite, {}).get(r.subject) == r.run_id]
        releases = [r for r in group if r.doc["kind"] == "release"]
        rep = pinned[0] if pinned else _newest(releases) or _newest(group)
        by = "pinned" if pinned else "latest-release" if releases else "latest"
        row = q.row(rep, metric, view, selected_by=by if per != "run" else None,
                    considered=len(group))
        values = [v for v in (r.value(metric, view) for r in group) if v is not None]
        row["n_runs"] = len(group)
        row["spread"] = {"min": min(values), "max": max(values)} if values else None
        row["_run"] = rep
        rows.append(row)
    rows.sort(key=lambda r: (r["registered_at"] or "", r["run_id"]))
    last: dict[tuple, Run] = {}
    for row in rows:
        run = row.pop("_run")
        prev = last.get((row["subject"], row["dataset"]))
        if prev is not None and any(prev.doc["scorer"].get(k) != run.doc["scorer"].get(k)
                                    for k in ("name", "version", "digest")):
            row["marker"] = "scorer changed"
        last[(row["subject"], row["dataset"])] = run
        if against:
            ref, how = _reference(q, run, against)
            if ref is None:
                row["against"] = None
            else:
                a, b = ref.value(metric, view), run.value(metric, view)
                row["against"] = {"run_id": ref.run_id, "how": how,
                                  "delta": (b - a) if None not in (a, b) else None,
                                  "param_changes": _summary(param_changes(ref.doc, run.doc))}
    args = {"engine": engine, "variant": variant, "dataset": dataset, "suite": suite,
            "per": per, "kind": ",".join(kinds) if kinds else None, "experiment": experiment,
            "metric": metric, "view": view, "against": against}
    return q.done("history", args, rows, limit=limit, fields=fields)


def _resolve_ref(q: Query, ref: str, dataset: str | None) -> Run:
    if "@" in ref:
        engine, _, version = ref.partition("@")
        q.check_engine(engine)
        runs = [r for r in q.all if r.engine == engine
                and r.doc["subject"]["engine_version"] == version]
        if not runs:
            versions = {r.doc["subject"]["engine_version"] for r in q.all if r.engine == engine}
            raise EvalError(f"no run of {engine} version {version!r}; versions: "
                            f"{_names(versions)}")
        names = {r.dataset for r in runs}
        if dataset is None:
            if len(names) != 1:
                raise EvalError(f"{ref} ran on {_names(names)}; pass --dataset")
            dataset = names.pop()
        runs = [r for r in runs if r.dataset == dataset]
        if not runs:
            raise EvalError(f"{ref} has no run on {dataset}")
        run, _by, _n = q.select(runs, "pinned", None, "all")
        run = run or _newest(q.included(runs))
        if run is None:
            raise EvalError(f"{ref}: every run on {dataset} is excluded (see --include)")
        return run
    rid = _core.check_run_id(ref)
    found = next((r for r in q.all if r.run_id == rid), None)
    if found is None:
        raise EvalError(f"run {rid} is not published in {q.ev.ns}", status=404)
    return found


def _common_dataset(q: Query, a: str, b: str, dataset: str | None) -> str | None:
    if dataset or ("@" not in a and "@" not in b):
        return dataset

    def names(ref: str) -> set[str]:
        if "@" in ref:
            engine, _, version = ref.partition("@")
            return {r.dataset for r in q.all if r.engine == engine
                    and r.doc["subject"]["engine_version"] == version}
        return {r.dataset for r in q.all if r.run_id == ref}

    both = names(a) & names(b)
    if len(both) == 1:
        return both.pop()
    if not both:
        raise EvalError(f"{a} and {b} share no dataset; pass --dataset")
    raise EvalError(f"{a} and {b} both ran on {_names(both)}; pass --dataset")


def _compare(q: Query, a: Run, b: Run) -> dict:
    deltas = []
    ma, mb = a.doc["metrics"], b.doc["metrics"]
    for view in sorted(set(ma) & set(mb)):
        for m in sorted(set(ma[view]) & set(mb[view])):
            va, vb = ma[view][m], mb[view][m]
            if va is None or vb is None:
                direction, delta = None, None
            else:
                delta = vb - va
                better = q.better(m)
                if sig6(va) == sig6(vb):
                    direction = "same"
                elif better is None:
                    direction = None
                else:
                    direction = "better" if (vb > va) == (better == "up") else "worse"
            deltas.append({"metric": m, "view": view, "a": va, "b": vb, "delta": delta,
                           "direction": direction})
    diffs = differences(a, b)
    if any(d["severity"] == "timing" for d in diffs) or any(
            q.metric_kind(d["metric"]) == "time" for d in deltas):
        q.unavailable.append(SLURM)
    ca, cb = a.doc["subject"].get("commit"), b.doc["subject"].get("commit")
    return {"a": q.row(a, q.primary(a), "all"), "b": q.row(b, q.primary(b), "all"),
            "differences": diffs, "comparable": comparable(diffs), "metric_deltas": deltas,
            "commit_range": f"{_core.sha7(ca)}..{_core.sha7(cb)}",
            "notes_a": _thread(q, a), "notes_b": _thread(q, b)}


def _thread(q: Query, run: Run) -> list[dict]:
    exp = run.doc.get("experiment")
    return [{k: n.get(k) for k in ("id", "at", "author", "tag", "text", "retracts")}
            for n in q.reg.notes
            if run.run_id in (n.get("runs") or []) or (exp and n.get("experiment") == exp)]


def compare(ev: Evals, a: str, b: str, *, dataset: str | None = None,
            fields: Sequence[str] | None = None) -> dict:
    q = Query(ev)
    dataset = _common_dataset(q, a, b, dataset)
    ra, rb = _resolve_ref(q, a, dataset), _resolve_ref(q, b, dataset)
    obj = _compare(q, ra, rb)
    obj["param_changes"] = _summary(param_changes(ra.doc, rb.doc))
    return q.done("compare", {"a": a, "b": b, "dataset": dataset}, [obj], fields=fields)


def diff(ev: Evals, a: str, b: str, *, fields: Sequence[str] | None = None) -> dict:
    q = Query(ev)
    ra, rb = _resolve_ref(q, a, None), _resolve_ref(q, b, None)
    obj = _compare(q, ra, rb)
    obj["param_changes"] = param_changes(ra.doc, rb.doc)

    def artefacts(run: Run) -> dict[str, str]:
        v = ev.ds.resolve(run.name)
        return {f.path: f.sha256 for f in v.files if f.path.startswith(ARTIFACTS + "/")}

    fa, fb = artefacts(ra), artefacts(rb)
    obj["artifacts"] = {"only_a": sorted(set(fa) - set(fb)), "only_b": sorted(set(fb) - set(fa)),
                        "changed": sorted(p for p in set(fa) & set(fb) if fa[p] != fb[p])}
    return q.done("diff", {"a": a, "b": b}, [obj], fields=fields)


def show(ev: Evals, ref: str, *, against: str | None = None,
         fields: Sequence[str] | None = None) -> dict:
    q = Query(ev)
    run = _resolve_ref(q, ref, None)
    ref_run, how = _reference(q, run, against)
    obj = {"run": run.doc, "name": run.name, "registered_at": run.registered_at,
           "notes_thread": _thread(q, run), "contamination": q.contamination(run, "all"),
           "carried": q.carried(run, "all"),
           "against": ref_run.run_id if ref_run else None, "against_how": how,
           "param_changes": param_changes(ref_run.doc, run.doc) if ref_run else None}
    if ev.backend == "gateway":
        try:
            obj["writer"] = ev.writer(run)
        except DatasetError as exc:
            q.unavailable.append({"what": "writer", "reason": exc.message})
    else:
        q.unavailable.append({"what": "writer", "reason": UNAVAILABLE})
    return q.done("show", {"run": ref, "against": against}, [obj], fields=fields)


def experiments(ev: Evals, *, engine: str | None = None, limit: int | None = None,
                fields: Sequence[str] | None = None) -> dict:
    q = Query(ev)
    runs = q.all
    if engine:
        q.check_engine(engine)
        runs = [r for r in runs if r.engine == engine]
    groups: dict[str, list[Run]] = {}
    for r in runs:
        if r.doc.get("experiment"):
            groups.setdefault(r.doc["experiment"], []).append(r)
    rows = [{"experiment": name, "n_runs": len(g), "subjects": sorted({r.subject for r in g}),
             "suites": sorted({r.suite for r in g}),
             "first": min(r.registered_at or "" for r in g) or None,
             "last": max(r.registered_at or "" for r in g) or None,
             "notes": sum(1 for n in q.reg.live_notes() if n.get("experiment") == name)}
            for name, g in sorted(groups.items())]
    rows.sort(key=lambda r: r["last"] or "", reverse=True)
    return q.done("experiments", {"engine": engine}, rows, limit=limit, fields=fields)


def experiment(ev: Evals, name: str, *, metric: str | None = None, view: str = "all",
               limit: int | None = None, fields: Sequence[str] | None = None) -> dict:
    q = Query(ev, view=view)
    runs = [r for r in q.all if r.doc.get("experiment") == name]
    if not runs:
        known = {r.doc.get("experiment") for r in q.all if r.doc.get("experiment")}
        raise EvalError(f"no run in experiment {name!r}; experiments: {_names(known)}")
    suites = sorted({r.suite for r in runs})
    q.check_view(suites)
    metrics = {s: q.metric_for(metric, {s}) for s in suites}
    rows: dict[tuple, dict] = {}
    for r in sorted(runs, key=lambda r: (r.registered_at or "", r.run_id)):
        key = (r.subject, r.doc["subject"]["config_digest"])
        row = rows.setdefault(key, {"subject": r.subject, "config_digest": key[1][:12],
                                    "cells": {}})
        m = metrics[r.suite]
        # The newest run of a (subject, config) in a suite is the cell; how
        # many there were is said, so a repeat is never silently dropped.
        earlier = row["cells"].get(r.suite)
        row["cells"][r.suite] = {"run_id": r.run_id, "metric": m, "value": r.value(m, view),
                                 "verdict": r.verdict, "status": r.doc["status"],
                                 "n_runs": (earlier["n_runs"] + 1) if earlier else 1,
                                 "notes": q.carried(r, view)}
    thread = [{k: n.get(k) for k in ("id", "at", "author", "tag", "text", "runs", "retracts")}
              for n in q.reg.notes if n.get("experiment") == name
              or set(n.get("runs") or []) & {r.run_id for r in runs}]
    return q.done("experiment", {"experiment": name, "metric": metric, "view": view},
                  list(rows.values()), limit=limit, fields=fields,
                  extra={"suites": suites, "experiment_notes": thread})


def search(ev: Evals, text: str, *, limit: int | None = None,
           fields: Sequence[str] | None = None) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise EvalError("search needs some text")
    q = Query(ev)
    needle = text.lower()
    rows = []
    for r in sorted(q.all, key=lambda r: (r.registered_at or "", r.run_id), reverse=True):
        for key in ("reason", "experiment", "notes"):
            value = r.doc.get(key)
            if isinstance(value, str) and needle in value.lower():
                rows.append({"type": "run", "run_id": r.run_id, "suite": r.suite,
                             "subject": r.subject, "registered_at": r.registered_at,
                             "field": key, "excerpt": _excerpt(value, needle)})
    for n in reversed(q.reg.notes):
        if needle in (n.get("text") or "").lower():
            rows.append({"type": "note", "id": n["id"], "at": n.get("at"), "tag": n.get("tag"),
                         "runs": n.get("runs"), "experiment": n.get("experiment"),
                         "excerpt": _excerpt(n["text"], needle)})
    return q.done("search", {"text": text}, rows, limit=limit, fields=fields)


def _excerpt(value: str, needle: str, width: int = 60) -> str:
    at = value.lower().find(needle)
    start = max(0, at - width // 2)
    piece = value[start:start + width + len(needle)].replace("\n", " ")
    return ("..." if start else "") + piece + ("..." if start + width + len(needle) < len(
        value) else "")


def coverage(ev: Evals, *, engine: str, limit: int | None = None,
             fields: Sequence[str] | None = None) -> dict:
    q = Query(ev)
    q.check_engine(engine)
    rows = []
    for s in sorted(set(q.suites) | {r.suite for r in q.all}):
        runs = [r for r in q.all if r.engine == engine and r.suite == s]
        wanted = ({_core.scored_input(q.suites[s])} if s in q.suites else set()) | {
            r.dataset for r in q.all if r.suite == s}
        ran = {r.dataset for r in runs}
        newest = _newest(runs)
        rows.append({"suite": s, "datasets_run": sorted(ran),
                     "datasets_not_run": sorted(wanted - ran), "n_runs": len(runs),
                     "newest_run": newest.run_id if newest else None,
                     "newest_at": newest.registered_at if newest else None})
    return q.done("coverage", {"engine": engine}, rows, limit=limit, fields=fields)


def parse_since(value: str | None) -> str:
    """`7d`, `12h`, or a date / time; returns the UTC time it means."""
    value = value or "7d"
    if value[:-1].isdigit() and value[-1:] in ("d", "h"):
        n = int(value[:-1])
        delta = timedelta(days=n) if value.endswith("d") else timedelta(hours=n)
        return _core.iso(_core.now() - delta)
    when = _core.parse_time(value if "T" in value else value + "T00:00:00Z")
    if when is None:
        raise EvalError("--since is <n>d, <n>h, or a date like 2026-10-01")
    return _core.iso(when)


def regressions(ev: Evals, *, since: str | None = None, scheduled_only: bool = False,
                limit: int | None = None, fields: Sequence[str] | None = None) -> dict:
    q = Query(ev)
    cut = parse_since(since)
    rows = []
    for r in sorted(q.all, key=lambda r: (r.registered_at or "", r.run_id), reverse=True):
        if (r.registered_at or "") < cut or r.verdict not in ("fail", "incomparable"):
            continue
        if scheduled_only and not _core.is_scheduled(r.doc):
            continue
        gate = r.doc["gate"]
        rows.append({"run_id": r.run_id, "suite": r.suite, "subject": r.subject,
                     "dataset": r.dataset, "registered_at": r.registered_at,
                     "verdict": r.verdict, "against": gate.get("against"),
                     "reason": gate.get("reason"), "failed": gate.get("failed") or [],
                     "kind": r.doc["kind"], "scheduled": _core.is_scheduled(r.doc),
                     "notes": q.carried(r, "all")})
    q.unavailable.append({"what": "alerts", "reason": UNAVAILABLE,
                          "detail": "open alert episodes live in the gateway's database"})
    return q.done("regressions", {"since": since or "7d", "scheduled_only": scheduled_only},
                  rows, limit=limit, fields=fields, extra={"since_utc": cut})


def catalog(ev: Evals, *, fields: Sequence[str] | None = None) -> dict:
    q = Query(ev)
    runs = q.all
    suites = {name: {"views": sorted(s["views"]), "primary": s["primary"],
                     "tie_break": s.get("tie_break") or [], "cadence": s.get("cadence"),
                     "dataset": _core.scored_input(s), "family": s.get("family")}
              for name, s in sorted(q.suites.items())}
    for r in runs:
        suites.setdefault(r.suite, {"registered": False})
    obj = {"generation": q.reg.generation,
           "engines": sorted({r.engine for r in runs}),
           "variants": sorted({r.variant for r in runs if r.variant}),
           "subjects": sorted({r.subject for r in runs}),
           "suites": suites, "datasets": sorted(q.datasets()),
           "metrics": [{k: m.get(k) for k in ("name", "kind", "better", "unit")}
                       for m in q.metrics.values()],
           "experiments": sorted({r.doc["experiment"] for r in runs if r.doc.get("experiment")}),
           "runs": len(runs), "pins": q.reg.current_pins()}
    return q.done("catalog", {}, [obj], fields=fields)


def _arg(name: str, help_: str, default=None) -> dict:
    return {"name": name, "help": help_, "default": default}


ROW_FIELDS = ["run_id", "suite", "dataset", "subject", "engine_version", "kind",
              "registered_at", "metric", "value", "verdict", "selected_by",
              "runs_considered", "comparable_to_leader", "notes"]
COMMON = [_arg("--evals", "the evaluation namespace <ns>-evals", "DATASETS_EVALS"),
          _arg("--url", "the backend", "DATASETS_URL"),
          _arg("--token-file", "a gateway credential, one line", TOKEN_VARIABLE),
          _arg("--limit", f"rows returned, at most {MAX_LIMIT}", DEFAULT_LIMIT),
          _arg("--fields", "keep only these row fields, comma-separated"),
          _arg("--table", "a table for people instead of JSON", False)]
SELECTION = [_arg("--select", " | ".join(SELECTS), "pinned"),
             _arg("--view", "a view the suite declares", "all"),
             _arg("--metric", "a registered metric", "the suite's primary"),
             _arg("--include", "bring back: " + ",".join(INCLUDABLE))]

VERBS = {
    "catalog": {"args": [], "rows": "one object: engines, variants, subjects, suites, "
                "datasets, metrics, experiments, generation, pins"},
    "scoreboard": {"args": [_arg("--datasets", "d,.."), _arg("--suites", "s,.."), *SELECTION],
                   "rows": "cells, one per (subject, dataset), state ok | no_value | "
                           "no_selection | not_run; plus coverage, common_subset, ranking "
                           "(each with comparable), unranked, ties, tradeoffs",
                   "fields": ROW_FIELDS + ["state", "shown"]},
    "best": {"args": [_arg("--dataset", "required"), *SELECTION],
             "rows": "ranked rows (exact, then scorer); input-different rows under `major`; "
                     "subjects with runs and none selectable under `no_selection`",
             "fields": ROW_FIELDS},
    "history": {"args": [_arg("--engine", "required"), _arg("--variant", "v"),
                         _arg("--dataset", "d"), _arg("--suite", "s"),
                         _arg("--per", "version | run | day", "version"),
                         _arg("--kind", "k,.."), _arg("--experiment", "x"),
                         _arg("--metric", "m"), _arg("--view", "v", "all"),
                         _arg("--against", "baseline | previous-in-experiment | <run>"),
                         _arg("--include", ",".join(INCLUDABLE))],
                "rows": "by registration time", "fields": ROW_FIELDS + [
                    "n_runs", "spread", "marker", "against"]},
    "compare": {"args": [_arg("a", "<run> | engine@version"),
                         _arg("b", "<run> | engine@version"), _arg("--dataset", "d")],
                "rows": "one object: a, b, differences, comparable, metric_deltas, "
                        "param_changes, commit_range, notes_a, notes_b"},
    "diff": {"args": [_arg("a", "<run>"), _arg("b", "<run>")],
             "rows": "as compare, with every differing param in full and artifacts "
                     "(only_a, only_b, changed)"},
    "show": {"args": [_arg("run", "<run> | engine@version"),
                      _arg("--against", "baseline | previous-in-experiment | <run>",
                           "baseline")],
             "rows": "one object: run (run.json as published), notes_thread, contamination, "
                     "carried, against, param_changes, writer"},
    "experiments": {"args": [_arg("--engine", "e")],
                    "rows": "experiment, n_runs, subjects, suites, first, last, notes"},
    "experiment": {"args": [_arg("name", "required"), _arg("--metric", "m"),
                            _arg("--view", "v", "all")],
                   "rows": "one per (subject, config): cells by suite, the newest run of "
                           "each with n_runs; plus experiment_notes"},
    "search": {"args": [_arg("text", "required")],
               "rows": "type run (field, excerpt) or note (id, tag, excerpt)"},
    "coverage": {"args": [_arg("--engine", "required")],
                 "rows": "suite, datasets_run, datasets_not_run, n_runs, newest_run"},
    "regressions": {"args": [_arg("--since", "<n>d | <n>h | date", "7d"),
                             _arg("--scheduled-only", "only scheduled runs", False)],
                    "rows": "verdict fail or incomparable, newest first"},
}
WRITES = {
    "record": "--out <root> --spec <file.json>: write run.json where the run executed",
    "publish": "<run-folder>...: push each run as <evals>/runs/<suite>/<run-id>",
    "init": "create registry generation g1",
    "suites register": "<file>: a new suites version when it changed",
    "metrics register": "<file>: a new metrics version when it changed",
    "pin": "<run-id> [--reason] [--suite]: append a pin",
    "note": "(--run <id>... | --experiment <x>) [--tag] [--retracts] <text>: append a note",
    "contamination add": "--kind --engine --dataset [--view] --since --reason [--ref]",
    "rehome": "start generation g<N+1> with the token in hand",
}


def help_json() -> dict:
    return {"schema": _core.QUERY_SCHEMA, "verb": "--help-json", "common": COMMON,
            "envelope": ["schema", "verb", "evals", "backend", "as_of", "args", "total",
                         "truncated", "rows", "notes", "excluded", "excluded_total",
                         "unavailable"],
            "verbs": VERBS, "writes": WRITES}
