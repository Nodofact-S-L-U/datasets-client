"""`python -m datasets_client evals <verb> ...`: JSON on stdout by default,
`--table` for people, `evals --help-json` for a session discovering the verbs.

Exit status: 0 done, 1 refused or failed (a JSON `error` on stdout and the
message on stderr), 2 a usage or configuration problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets_client._client import Datasets
from datasets_client.errors import ConfigError, DatasetError
from datasets_client.evals import _query
from datasets_client.evals._core import EvalError, parse_time
from datasets_client.evals._record import record
from datasets_client.evals._store import Evals


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datasets_client evals",
                                     description="evaluation runs: record, publish, "
                                                 "register, query")
    parser.add_argument("--help-json", action="store_true",
                        help="every verb, its arguments, defaults and row fields, as JSON")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=None, help="the backend (default: DATASETS_URL)")
    common.add_argument("--token-file", default=None,
                        help="a gateway credential, one line (default: DATASETS_TOKEN)")
    common.add_argument("--evals", default=None,
                        help="the evaluation namespace <ns>-evals (default: DATASETS_EVALS)")
    common.add_argument("--table", action="store_true", help="a table instead of JSON")
    common.add_argument("--limit", type=int, default=None)
    common.add_argument("--fields", default=None, help="keep only these row fields")
    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument("--metric", default=None)
    sel.add_argument("--select", default="pinned", choices=_query.SELECTS)
    sel.add_argument("--view", default="all")
    sel.add_argument("--include", default=None,
                     help="bring back: " + ",".join(_query.INCLUDABLE))
    sub = parser.add_subparsers(dest="verb")

    rec = sub.add_parser("record", parents=[common], help="write run.json where it ran")
    rec.add_argument("--out", required=True, help="the output root; the run folder goes in it")
    rec.add_argument("--spec", required=True,
                     help="JSON: everything `datasets_client.evals.record` takes")
    pub = sub.add_parser("publish", parents=[common], help="publish run folders")
    pub.add_argument("folders", nargs="+")
    sub.add_parser("init", parents=[common], help="create registry generation g1")
    sub.add_parser("rehome", parents=[common], help="start generation g<N+1>")
    for kind in ("suites", "metrics"):
        reg = sub.add_parser(kind, help=f"the {kind} registry")
        regsub = reg.add_subparsers(dest="action", required=True)
        r = regsub.add_parser("register", parents=[common])
        r.add_argument("file")
    pin = sub.add_parser("pin", parents=[common], help="pin a run as its subject's baseline")
    pin.add_argument("run_id")
    pin.add_argument("--reason", default=None)
    pin.add_argument("--suite", default=None)
    pin.add_argument("--author", default=None)
    note = sub.add_parser("note", parents=[common], help="append a note")
    note.add_argument("text")
    note.add_argument("--run", action="append", default=[], dest="runs")
    note.add_argument("--experiment", default=None)
    note.add_argument("--tag", default=None)
    note.add_argument("--retracts", default=None)
    note.add_argument("--author", default=None)
    cont = sub.add_parser("contamination", help="contamination records")
    contsub = cont.add_subparsers(dest="action", required=True)
    add = contsub.add_parser("add", parents=[common])
    add.add_argument("--kind", required=True)
    add.add_argument("--engine", required=True)
    add.add_argument("--dataset", required=True)
    add.add_argument("--view", default=None)
    add.add_argument("--since", required=True)
    add.add_argument("--reason", required=True)
    add.add_argument("--ref", default=None)
    add.add_argument("--author", default=None)

    sub.add_parser("catalog", parents=[common])
    sb = sub.add_parser("scoreboard", parents=[common, sel])
    sb.add_argument("--datasets", default=None)
    sb.add_argument("--suites", default=None)
    b = sub.add_parser("best", parents=[common, sel])
    b.add_argument("--dataset", required=True)
    h = sub.add_parser("history", parents=[common])
    h.add_argument("--engine", required=True)
    h.add_argument("--variant", default=None)
    h.add_argument("--dataset", default=None)
    h.add_argument("--suite", default=None)
    h.add_argument("--per", default="version", choices=("version", "run", "day"))
    h.add_argument("--kind", default=None)
    h.add_argument("--experiment", default=None)
    h.add_argument("--metric", default=None)
    h.add_argument("--view", default="all")
    h.add_argument("--against", default=None)
    h.add_argument("--include", default=None)
    c = sub.add_parser("compare", parents=[common])
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--dataset", default=None)
    d = sub.add_parser("diff", parents=[common])
    d.add_argument("a")
    d.add_argument("b")
    s = sub.add_parser("show", parents=[common])
    s.add_argument("run")
    s.add_argument("--against", default=None)
    e = sub.add_parser("experiments", parents=[common])
    e.add_argument("--engine", default=None)
    x = sub.add_parser("experiment", parents=[common])
    x.add_argument("name")
    x.add_argument("--metric", default=None)
    x.add_argument("--view", default="all")
    se = sub.add_parser("search", parents=[common])
    se.add_argument("text")
    cv = sub.add_parser("coverage", parents=[common])
    cv.add_argument("--engine", required=True)
    rg = sub.add_parser("regressions", parents=[common])
    rg.add_argument("--since", default=None)
    rg.add_argument("--scheduled-only", action="store_true")
    return parser


def _evals(args) -> Evals:
    from datasets_client._cli import read_token

    return Evals(args.evals, ds=Datasets(args.url, token=read_token(args.token_file)))


def _record(args) -> dict:
    try:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read the record spec {args.spec}: {exc}") from None
    if not isinstance(spec, dict):
        raise ConfigError(f"{args.spec}: a record spec is a JSON object")
    if "started_at" in spec:
        when = parse_time(spec["started_at"])
        if when is None:
            raise EvalError("spec.started_at is a UTC time")
        spec["started_at"] = when
    from datasets_client._cli import read_token

    spec.setdefault("ds", Datasets(args.url, token=read_token(args.token_file)))
    folder = record(args.out, **spec)
    return {"run_folder": str(folder), "run_id": folder.name}


def run(args) -> dict:
    verb = args.verb
    if verb == "record":
        return _record(args)
    ev = _evals(args)
    limit, fields = args.limit, _csv(args.fields)
    include = _csv(getattr(args, "include", None)) or ()
    if verb == "publish":
        return {"published": [ev.publish(f) for f in args.folders]}
    if verb == "init":
        return ev.init()
    if verb == "rehome":
        return ev.rehome()
    if verb == "suites":
        return ev.register_suites(args.file)
    if verb == "metrics":
        return ev.register_metrics(args.file)
    if verb == "pin":
        return ev.pin(args.run_id, reason=args.reason, author=args.author, suite=args.suite)
    if verb == "note":
        return ev.note(args.text, runs=args.runs, experiment=args.experiment, tag=args.tag,
                       retracts=args.retracts, author=args.author)
    if verb == "contamination":
        return ev.contaminate(kind=args.kind, engine=args.engine, dataset=args.dataset,
                              since=args.since, reason=args.reason, view=args.view,
                              ref=args.ref, author=args.author)
    if verb == "catalog":
        return _query.catalog(ev, fields=fields)
    if verb == "scoreboard":
        return _query.scoreboard(ev, metric=args.metric, datasets=_csv(args.datasets),
                                 suites=_csv(args.suites), select=args.select, view=args.view,
                                 include=include, limit=limit, fields=fields)
    if verb == "best":
        return _query.best(ev, dataset=args.dataset, metric=args.metric, select=args.select,
                           view=args.view, include=include, limit=limit, fields=fields)
    if verb == "history":
        return _query.history(ev, engine=args.engine, variant=args.variant,
                              dataset=args.dataset, suite=args.suite, per=args.per,
                              kinds=_csv(args.kind), experiment=args.experiment,
                              metric=args.metric, view=args.view, against=args.against,
                              include=include, limit=limit, fields=fields)
    if verb == "compare":
        return _query.compare(ev, args.a, args.b, dataset=args.dataset, fields=fields)
    if verb == "diff":
        return _query.diff(ev, args.a, args.b, fields=fields)
    if verb == "show":
        return _query.show(ev, args.run, against=args.against, fields=fields)
    if verb == "experiments":
        return _query.experiments(ev, engine=args.engine, limit=limit, fields=fields)
    if verb == "experiment":
        return _query.experiment(ev, args.name, metric=args.metric, view=args.view,
                                 limit=limit, fields=fields)
    if verb == "search":
        return _query.search(ev, args.text, limit=limit, fields=fields)
    if verb == "coverage":
        return _query.coverage(ev, engine=args.engine, limit=limit, fields=fields)
    if verb == "regressions":
        return _query.regressions(ev, since=args.since, scheduled_only=args.scheduled_only,
                                  limit=limit, fields=fields)
    raise ConfigError(f"unknown verb {verb!r}")


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)[:60]
    return str(value)


def table(answer: dict) -> str:
    """Rows as aligned columns; a one-object answer as `key: value` lines."""
    rows = answer.get("rows") if isinstance(answer, dict) else None
    if rows is None:
        return json.dumps(answer, indent=1, ensure_ascii=False)
    lines = []
    if len(rows) == 1 and answer.get("total") == 1 and "run_id" not in rows[0]:
        lines += [f"{k}: {_cell(v)}" for k, v in rows[0].items()]
    elif rows:
        # Every row's keys, first seen first: a scoreboard's first cell may be
        # `not_run`, which has no value or run id to show.
        cols = [k for k in dict.fromkeys(k for r in rows for k in r) if k != "notes"]
        cells = [[_cell(r.get(k)) for k in cols] for r in rows]
        widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
        lines.append("  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True)))
        lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True))
                  for row in cells]
    else:
        lines.append("(no rows)")
    for key in ("ranking", "tradeoffs", "major", "no_selection"):
        if answer.get(key):
            lines.append(f"{key}: {_cell(answer[key])}")
    lines.append(f"total {answer.get('total')}" + (" (truncated)" if answer.get("truncated")
                                                   else ""))
    lines += [f"note: {n}" for n in answer.get("notes", [])]
    if answer.get("excluded_total"):
        lines.append(f"excluded: {answer['excluded_total']} "
                     f"({', '.join(e['run_id'] + ' ' + e['why'] for e in answer['excluded'])})")
    lines += [f"unavailable: {u['what']}: {u['reason']}" for u in answer.get("unavailable", [])]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.help_json:
        print(json.dumps(_query.help_json(), indent=1, ensure_ascii=False))
        return 0
    if not args.verb:
        parser.print_usage(sys.stderr)
        return 2
    try:
        answer = run(args)
    except ConfigError as exc:
        print(f"datasets_client evals: {exc.message}", file=sys.stderr)
        print(json.dumps({"error": {"type": "ConfigError", "message": exc.message}}))
        return 2
    except DatasetError as exc:
        print(f"datasets_client evals: {type(exc).__name__}: {exc.message}", file=sys.stderr)
        print(json.dumps({"error": {"type": type(exc).__name__, "message": exc.message,
                                    "status": exc.status}}, ensure_ascii=False))
        return 1
    if getattr(args, "table", False):
        print(table(answer))
    else:
        print(json.dumps(answer, indent=1, ensure_ascii=False))
    return 0
