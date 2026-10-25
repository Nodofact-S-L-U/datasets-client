"""`python -m datasets_client` (also `datasets-client`): ls, pull, push, copy, evals.

Every command takes `--url` (default `DATASETS_URL`) and, for a gateway,
`--token-file`. `pull` and `push` are copies between a backend and a local
directory; `copy` is between any two backends - loading a Blob
container into a gateway, or the reverse, is one command.

Written in Python rather than shell on purpose: Git Bash rewrites any argument
that looks like `/a/path`, and a dataset name is exactly that shape.

`evals <verb> ...` is the evaluation layer (`datasets_client.evals`), with its
own arguments: `evals --help`, or `evals --help-json` for every verb as JSON.

Exit status: 0 done, 1 refused or failed (the named error is printed), 2 a
usage or configuration problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets_client import __version__
from datasets_client._client import Datasets, copy
from datasets_client._layout import parse_ref
from datasets_client.errors import ConfigError, DatasetError, Refused


def read_token(path: str | None) -> str | None:
    if not path:
        return None
    try:
        token = Path(path).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"cannot read the token file {path}: {exc}") from exc
    if not token or "\n" in token:
        raise ConfigError(f"{path} must hold exactly one token on one line")
    return token


def read_metadata(path: str | None) -> dict | None:
    if not path:
        return None
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read metadata from {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: metadata is a JSON object")
    return doc


def human(n: int | None) -> str:
    if n is None:
        return "-"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def print_listing(rows) -> None:
    if not rows:
        print("(no datasets under that prefix)")
        return
    for r in rows:
        if r.moved_to:
            print(f"{r.name}  -> moved to {r.moved_to}")
            continue
        print(f"{r.name}  v{r.latest_version}  {r.file_count} files  {human(r.total_bytes)}"
              + (f"  {r.metadata.get('description')}" if r.metadata.get("description") else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datasets_client", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=__version__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=None, help="the backend (default: DATASETS_URL)")
    common.add_argument("--token-file", default=None,
                        help="a gateway credential, one line (default: DATASETS_TOKEN)")
    sub = parser.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("ls", parents=[common], help="list datasets")
    ls.add_argument("prefix", nargs="?", default="")

    pull = sub.add_parser("pull", parents=[common],
                          help="copy name[@v] into a local folder, in the layout")
    pull.add_argument("ref")
    pull.add_argument("dest")

    push = sub.add_parser("push", parents=[common],
                          help="a local directory as a dataset, or its next version")
    push.add_argument("source")
    push.add_argument("name")
    push.add_argument("--parent", type=int, default=None,
                      help="the version this builds on (required for a new version)")
    push.add_argument("--metadata-file", default=None,
                      help="JSON: description, source, ci_allowed, data_class "
                           "(synthetic; public + licence; client + approved_by) - "
                           "required for a new dataset")
    push.add_argument("--label", default=None)
    push.add_argument("--breaking", action="store_true",
                      help="the file set or a CSV header changed, on purpose")

    ab = sub.add_parser("abort", parents=[common],
                        help="release an open upload, so its quota frees now (gateway)")
    ab.add_argument("upload_id")

    cp = sub.add_parser("copy", help="copy name[@v] from one backend to another")
    cp.add_argument("ref")
    cp.add_argument("--from", dest="src", required=True)
    cp.add_argument("--to", dest="dst", required=True)
    cp.add_argument("--from-token-file", default=None)
    cp.add_argument("--to-token-file", default=None)
    cp.add_argument("--breaking", action="store_true")

    ev = sub.add_parser("evals", add_help=False,
                        help="evaluation runs: record, publish, register, query "
                             "(evals --help)")
    ev.add_argument("rest", nargs=argparse.REMAINDER)
    return parser


def run(args) -> int:
    if args.command == "evals":
        from datasets_client.evals._cli import main as evals_main
        return evals_main(args.rest)
    if args.command == "copy":
        name, version = parse_ref(args.ref)
        v = copy(Datasets(args.src, token=read_token(args.from_token_file)),
                 Datasets(args.dst, token=read_token(args.to_token_file)),
                 name, version, breaking=args.breaking)
        print(f"copied {v.name}@{v.version} ({len(v.files)} files) to {args.dst}")
        return 0

    ds = Datasets(args.url, token=read_token(args.token_file))
    if args.command == "ls":
        print_listing(ds.list(args.prefix))
    elif args.command == "pull":
        v = ds.pull(args.ref, args.dest)
        where = Path(args.dest).joinpath(*v.name.split("/"), f"v{v.version}")
        print(f"pulled {v.name}@{v.version} ({len(v.files)} files) into {where}")
    elif args.command == "push":
        got = ds.push(args.source, args.name, parent=args.parent,
                      metadata=read_metadata(args.metadata_file), label=args.label,
                      breaking=args.breaking)
        print(f"published {got['name']}@{got['version']}")
    elif args.command == "abort":
        got = ds.abort(args.upload_id)
        print(f"aborted {got.get('upload_id', args.upload_id)}: its quota is free now")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["evals"]:
        # Its own parser, whole: argparse would otherwise claim `--help` here.
        from datasets_client.evals._cli import main as evals_main
        return evals_main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ConfigError as exc:
        print(f"datasets_client: {exc.message}", file=sys.stderr)
        return 2
    except DatasetError as exc:
        print(f"datasets_client: {type(exc).__name__}: {exc.message}", file=sys.stderr)
        for p in exc.problems if isinstance(exc, Refused) else []:
            print(f"  {p.get('path')}: {p.get('problem')}", file=sys.stderr)
        return 1
