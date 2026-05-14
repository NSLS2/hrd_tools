"""
``python -m hrd_tools.tiled.cli`` -- command-line entry points.

Two subcommands:

``register``
    Walk a source filesystem root for v3 datasets, rewrite each into a
    per-detector / per-chunk parquet cache, and register the result
    against a running tiled server.

``init-config``
    Emit a tiled-server config YAML that points at a specific source
    root, cache root, and catalog database.  The generated file is
    consumed by ``tiled serve config <out>``.

The two roots are kept distinct because the source filesystem is
treated as read-only and the cache is owned by us; both must appear in
``readable_storage`` so the tiled server can resolve the ``file://``
URIs that are written into catalog rows.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from hrd_tools.tiled import LAYOUT

__all__ = ["build_parser", "main"]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_register_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "root",
        type=Path,
        help="Filesystem root to walk for v3 directories.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help=(
            "Required.  Directory under which to place the rewritten"
            " parquet cache.  Must be distinct from <root>."
        ),
    )
    parser.add_argument(
        "--tiled-uri",
        default="http://localhost:8000",
        help="URI of the tiled server.  Default: %(default)s",
    )
    parser.add_argument(
        "--tiled-api-key",
        default=None,
        help=(
            "API key for the tiled server.  Falls back to the TILED_API_KEY"
            " environment variable if not given."
        ),
    )
    parser.add_argument(
        "--catalog-path",
        default="/",
        help=(
            "Path within the tiled catalog under which to mount the"
            " walked tree.  Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=LAYOUT.default_frames_per_chunk,
        help="Frame-axis chunk size for the cache (default: %(default)s).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Delete an existing run node before re-registering.  Without"
            " this flag, attempting to register a run that already exists"
            " is an error."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )


def _add_init_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Filesystem root containing v3 datasets.  Will be added to"
            " readable_storage in the generated config."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help=(
            "Directory under which the rewrite step places its parquet"
            " cache.  Will be added to readable_storage in the generated"
            " config.  Must be distinct from --source-root and not nested"
            " inside it (or vice versa)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("config.yml"),
        help="Output YAML file path (default: %(default)s).",
    )
    parser.add_argument(
        "--catalog-db",
        type=Path,
        default=Path("catalog.db"),
        help=(
            "Catalog sqlite database path (default: %(default)s,"
            " resolved at invocation time)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hrd_tools.tiled.cli",
        description=(
            "Tools for the multihead v3 tiled integration.  Use the"
            " ``init-config`` subcommand to generate a tiled-server"
            " config YAML, and the ``register`` subcommand to populate"
            " a running tiled server from a v3 source tree."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser(
        "register",
        help="Register v3 datasets with a running tiled server.",
        description=(
            "Register every multihead v3 dataset under <root> with a tiled"
            " server.  Each dataset is rewritten into a per-detector,"
            " per-chunk parquet cache under --cache-root; the cache layout"
            " is consumed directly by tiled's built-in adapters."
        ),
    )
    _add_register_args(register)

    init_config = subparsers.add_parser(
        "init-config",
        help="Generate a tiled-server config YAML for a v3 cache.",
        description=(
            "Write a tiled-server config YAML that points at the given"
            " source root, cache root, and catalog database.  The point"
            " of this subcommand is to capture deployment information"
            " (the absolute paths) that the tooling cannot guess."
        ),
    )
    _add_init_config_args(init_config)

    return parser


# ---------------------------------------------------------------------------
# Subcommand: register
# ---------------------------------------------------------------------------


def _connect(uri: str, api_key: str | None, catalog_path: str) -> object:
    from tiled.client import from_uri  # local import: optional dep

    key = api_key if api_key is not None else os.environ.get("TILED_API_KEY")
    client = from_uri(uri, api_key=key)
    # Navigate to catalog_path -- e.g. ``/`` is the root client itself.
    path_parts = [p for p in catalog_path.split("/") if p]
    for part in path_parts:
        client = client[part]
    return client


def _cmd_register(args: argparse.Namespace, log: logging.Logger) -> int:
    from hrd_tools.tiled.walker import register_root  # local import: optional dep

    root: Path = args.root.resolve()
    cache_root: Path = args.cache_root.resolve()

    if not root.is_dir():
        log.error("root %s is not a directory", root)
        return 2
    if cache_root == root:
        log.error("--cache-root must differ from <root>")
        return 2
    try:
        cache_root.relative_to(root)
        log.error("--cache-root must not be inside <root>")
        return 2
    except ValueError:
        pass
    try:
        root.relative_to(cache_root)
        log.error("<root> must not be inside --cache-root")
        return 2
    except ValueError:
        pass

    cache_root.mkdir(parents=True, exist_ok=True)

    client = _connect(args.tiled_uri, args.tiled_api_key, args.catalog_path)
    log.info("connected to %s at %s", args.tiled_uri, args.catalog_path)

    registered = register_root(
        client,
        root,
        cache_root,
        frames_per_chunk=args.frames_per_chunk,
        replace=args.replace,
    )
    log.info("registered %d v3 dataset(s) under %s", len(registered), root)
    for src in registered:
        log.info("  %s", src)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: init-config
# ---------------------------------------------------------------------------


def _check_disjoint(a: Path, b: Path, *, a_label: str, b_label: str) -> str | None:
    """
    Return an error message if *a* and *b* nest in either direction,
    else ``None``.
    """
    if a == b:
        return f"{a_label} and {b_label} must differ ({a})"
    try:
        a.relative_to(b)
        return f"{a_label} ({a}) must not be inside {b_label} ({b})"
    except ValueError:
        pass
    try:
        b.relative_to(a)
        return f"{b_label} ({b}) must not be inside {a_label} ({a})"
    except ValueError:
        pass
    return None


def _build_config(
    *,
    source_root: Path,
    cache_root: Path,
    catalog_db: Path,
) -> dict[str, Any]:
    """Return the parsed-YAML dict shape for a tiled-server config."""
    return {
        "authentication": {
            "single_user_api_key": "${TILED_API_KEY}",
            "allow_anonymous_access": True,
        },
        "trees": [
            {
                "tree": "catalog",
                "path": "/",
                "args": {
                    "uri": f"sqlite:///{catalog_db}",
                    "readable_storage": [str(source_root), str(cache_root)],
                    "init_if_not_exists": True,
                },
            }
        ],
    }


def _cmd_init_config(args: argparse.Namespace, log: logging.Logger) -> int:
    source_root: Path = args.source_root.resolve()
    cache_root: Path = args.cache_root.resolve()
    catalog_db: Path = args.catalog_db.resolve()
    out: Path = args.out.resolve()

    err = _check_disjoint(
        source_root, cache_root, a_label="--source-root", b_label="--cache-root"
    )
    if err is not None:
        log.error("%s", err)
        return 2

    if out.exists() and not args.force:
        log.error("%s already exists; pass --force to overwrite", out)
        return 2

    # Create parent dirs for the output and the catalog db (the catalog
    # db itself will be created by tiled on first run because of
    # ``init_if_not_exists: true``).
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog_db.parent.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        source_root=source_root,
        cache_root=cache_root,
        catalog_db=catalog_db,
    )
    with out.open("w") as fout:
        yaml.safe_dump(config, fout, sort_keys=False)

    log.info("wrote tiled config to %s", out)
    log.info("  source_root  = %s", source_root)
    log.info("  cache_root   = %s", cache_root)
    log.info("  catalog db   = %s", catalog_db)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("hrd_tools.tiled.cli")

    if args.command == "register":
        return _cmd_register(args, log)
    if args.command == "init-config":
        return _cmd_init_config(args, log)
    # argparse should have already rejected unknown subcommands.
    log.error("unknown subcommand %r", args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
