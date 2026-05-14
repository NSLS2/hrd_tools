"""
Discover v3 data directories under a root and register them with tiled.

The walker mirrors the source filesystem structure under a tiled
container tree.  Every directory beneath ``root`` becomes a container
(creating intermediate containers as needed); any directory that is
itself a v3 dataset (see
:func:`hrd_tools.tiled.rewrite.is_v3_directory`) additionally gets a
``primary`` (and optionally ``baseline``) child container in the
bluesky-shaped layout:

.. code-block:: text

    <run-key>/                          (container, run metadata)
        primary/                        (container)
            det_01 ... det_NN           (sparse, (n_frames, n_rows, n_cols))
            scalars                     (table)
        baseline/                       (container, optional)
            readings                    (table)

The detector count, frame count, pixel grid, and dtype are *discovered*
from the rewrite result rather than hardcoded.

Because the rewrite step produces parquet files in exactly the layout
tiled's built-in :class:`SparseBlocksParquetAdapter` and
:class:`ParquetDatasetAdapter` expect, the only runtime customization
needed on the server is the mimetype-to-adapter mapping in the tiled
config.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tiled.structures.array import BuiltinDtype
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management
from tiled.structures.sparse import COOStructure
from tiled.structures.table import TableStructure
from tiled.utils import ensure_uri

from hrd_tools.tiled import LAYOUT
from hrd_tools.tiled.rewrite import (
    RewriteResult,
    is_v3_directory,
    rewrite_v3_directory,
)

__all__ = [
    "build_run_metadata",
    "discover_v3_directories",
    "register_root",
    "register_run",
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_v3_directories(root: Path) -> Iterator[Path]:
    """
    Yield, in lexicographic order, every directory under *root* that is
    itself a v3 dataset.

    *root* itself may be a v3 directory and will be yielded if so.
    Hidden directories (names starting with ``.``) are skipped.
    """
    root = Path(root)
    if not root.is_dir():
        return

    if is_v3_directory(root):
        yield root
        # Do not descend into a v3 directory's interior.
        return

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        yield from discover_v3_directories(entry)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _load_metadata_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r") as fin:
        return json.load(fin)


def _nominal_bin_from_scalars(scalars_path: Path) -> float | None:
    md = pq.read_schema(scalars_path).metadata or {}
    raw = md.get(LAYOUT.scalars_nominal_bin_metadata_key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _jsonify(value: Any) -> Any:
    """
    Convert a Python value to something safe to send through tiled's
    JSON metadata channel.  Handles dataclasses, numpy scalars, bytes,
    and nested containers.
    """
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonify(asdict(value))
    return value


def build_run_metadata(
    source_dir: Path,
    rewrite: RewriteResult,
) -> dict[str, Any]:
    """
    Assemble the metadata dict attached to the top-level run container.

    This is the union of:

    - format-version markers (``format_version``, ``n_frames``,
      ``shape``, ``stream_names``),
    - ``nominal_bin`` lifted from the scalars schema,
    - everything in ``metadata.json`` (``scan_md``, ``scan_config``,
      ``staff_log``, ...).
    """
    nominal_bin = _nominal_bin_from_scalars(rewrite.scalars_path)
    extra = _load_metadata_json(rewrite.metadata_json_path)
    stream_names = ["primary"]
    if rewrite.baseline_path is not None:
        stream_names.append("baseline")

    md: dict[str, Any] = {
        "format_version": LAYOUT.format_version,
        "source_path": str(source_dir.resolve()),
        "n_frames": rewrite.n_frames,
        "shape": list(rewrite.shape),
        "chunks": list(rewrite.chunks),
        "stream_names": stream_names,
    }
    if nominal_bin is not None:
        md["nominal_bin"] = nominal_bin
    md.update(_jsonify(extra))
    return md


# ---------------------------------------------------------------------------
# DataSource construction
# ---------------------------------------------------------------------------


def _detector_data_source(
    block_paths: list[Path],
    chunks: tuple[int, ...],
    data_dtype: np.dtype,
    *,
    n_rows: int,
    n_cols: int,
) -> DataSource[COOStructure]:
    """Build the sparse DataSource for a single detector."""
    if len(block_paths) != len(chunks):
        raise ValueError(
            f"Detector block count {len(block_paths)} does not match"
            f" chunk count {len(chunks)}"
        )

    structure = COOStructure(
        chunks=(chunks, (n_rows,), (n_cols,)),
        shape=(sum(chunks), n_rows, n_cols),
        data_type=BuiltinDtype.from_numpy_dtype(np.dtype(data_dtype)),
        dims=("frame", "row", "col"),
    )

    assets = [
        Asset(
            data_uri=ensure_uri(str(path.resolve())),
            is_directory=False,
            parameter="data_uris",
            num=i,
        )
        for i, path in enumerate(block_paths)
    ]
    return DataSource(
        structure_family=StructureFamily.sparse,
        structure=structure,
        mimetype=LAYOUT.parquet_sparse_mimetype,
        assets=assets,
        management=Management.external,
    )


def _table_data_source(parquet_path: Path) -> DataSource[TableStructure]:
    """Build a single-partition table DataSource from a parquet file."""
    schema = pq.read_schema(parquet_path)
    structure = TableStructure.from_schema(schema, npartitions=1)
    asset = Asset(
        data_uri=ensure_uri(str(parquet_path.resolve())),
        is_directory=False,
        parameter="data_uris",
        num=0,
    )
    return DataSource(
        structure_family=StructureFamily.table,
        structure=structure,
        mimetype=LAYOUT.parquet_table_mimetype,
        assets=[asset],
        management=Management.external,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _get_or_create_container(
    client: Any,
    key: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """
    Return ``client[key]`` if it already exists, otherwise create a new
    container child with the given metadata.
    """
    try:
        existing = client[key]
    except KeyError:
        return client.create_container(key=key, metadata=metadata or {})
    return existing


def _ensure_parent_path(client: Any, parts: tuple[str, ...]) -> Any:
    """
    Descend from ``client`` creating intermediate containers as needed.
    Returns the deepest container.  ``parts`` may be empty.
    """
    node = client
    for part in parts:
        node = _get_or_create_container(node, part)
    return node


def _replace_child(parent: Any, key: str) -> None:
    """Delete child *key* under *parent* if it exists."""
    try:
        child = parent[key]
    except KeyError:
        return
    child.delete(recursive=True, external_only=True)


def register_run(
    parent: Any,
    run_key: str,
    source_dir: Path,
    cache_dir: Path,
    *,
    frames_per_chunk: int = LAYOUT.default_frames_per_chunk,
    replace: bool = False,
) -> Any:
    """
    Rewrite a single v3 dataset and register it under *parent*.

    Parameters
    ----------
    parent
        A tiled container client (e.g. the root client or a nested
        container) that will host the run node.
    run_key
        Key for the new run container inside *parent*.
    source_dir
        Path to the v3 dataset directory.
    cache_dir
        Output directory for the rewritten parquet cache.
    frames_per_chunk
        Frame-axis chunk size passed to
        :func:`hrd_tools.tiled.rewrite.rewrite_v3_directory`.
    replace
        When ``True``, delete an existing ``run_key`` under *parent*
        before creating the new one.

    Returns
    -------
    The newly created run container client.
    """
    rewrite = rewrite_v3_directory(
        Path(source_dir), Path(cache_dir), frames_per_chunk=frames_per_chunk
    )

    run_metadata = build_run_metadata(Path(source_dir), rewrite)

    if replace:
        _replace_child(parent, run_key)

    run_node = parent.create_container(key=run_key, metadata=run_metadata)

    n_detectors = rewrite.n_detectors
    n_rows = rewrite.n_rows
    n_cols = rewrite.n_cols

    # ---- primary stream --------------------------------------------------
    primary = run_node.create_container(
        key="primary",
        metadata={"stream_name": "primary"},
    )
    for det in range(1, n_detectors + 1):
        ds = _detector_data_source(
            rewrite.detector_block_paths[det],
            rewrite.chunks,
            rewrite.data_dtype,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        primary.new(
            structure_family=ds.structure_family,
            data_sources=[ds],
            key=f"det_{det:02d}",
            metadata={"detector": det},
        )
    scalars_ds = _table_data_source(rewrite.scalars_path)
    primary.new(
        structure_family=scalars_ds.structure_family,
        data_sources=[scalars_ds],
        key="scalars",
        metadata={},
    )

    # ---- baseline stream -------------------------------------------------
    if rewrite.baseline_path is not None:
        baseline = run_node.create_container(
            key="baseline",
            metadata={"stream_name": "baseline"},
        )
        baseline_ds = _table_data_source(rewrite.baseline_path)
        baseline.new(
            structure_family=baseline_ds.structure_family,
            data_sources=[baseline_ds],
            key="readings",
            metadata={},
        )

    return run_node


def register_root(
    client: Any,
    root: Path,
    cache_root: Path,
    *,
    frames_per_chunk: int = LAYOUT.default_frames_per_chunk,
    replace: bool = False,
) -> list[Path]:
    """
    Discover every v3 directory under *root* and register each one,
    mirroring the relative path under *client* via nested containers.

    Parameters
    ----------
    client
        Tiled container client to use as the registration root.
    root
        Filesystem root to walk.
    cache_root
        Directory under which rewritten parquet caches are placed.  The
        cache for a run at ``<root>/a/b/c`` lives at
        ``<cache_root>/a/b/c``.
    frames_per_chunk
        Frame-axis chunk size.
    replace
        When ``True``, delete each pre-existing run node before
        re-registering.

    Returns
    -------
    list[Path]
        The source directories that were registered, in order.
    """
    root = Path(root).resolve()
    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    registered: list[Path] = []
    for src in discover_v3_directories(root):
        rel = src.resolve().relative_to(root)
        cache_dir = cache_root / rel
        # The deepest path component is the run key; intermediate
        # components are nested containers.
        parts = rel.parts
        if not parts:
            # ``root`` itself is a v3 dir.  Use its basename as the run
            # key under the catalog root.
            parent = client
            run_key = root.name or "run"
        else:
            parent = _ensure_parent_path(client, parts[:-1])
            run_key = parts[-1]
        register_run(
            parent,
            run_key,
            src,
            cache_dir,
            frames_per_chunk=frames_per_chunk,
            replace=replace,
        )
        registered.append(src)
    return registered
