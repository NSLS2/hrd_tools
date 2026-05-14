"""
Tiled integration for the multihead v3 sparse data format.

This subpackage provides:

- :class:`V3Layout` / :data:`LAYOUT` -- file-name, mimetype, and
  format-version constants that are *invariant* across datasets.  Any
  values discoverable from the data itself (n_detectors, n_frames,
  pixel grid, dtype, ...) are **not** carried here; they flow through
  :class:`hrd_tools.tiled.rewrite.RewriteResult`.
- :mod:`hrd_tools.tiled.rewrite` -- streaming converter that takes a
  v3 data directory (``images*.parquet``, ``scalars*.parquet``,
  ``metadata.json``, ``baseline.parquet``) and produces a tiled-friendly
  per-detector, per-chunk cache under a separate cache root.
- :mod:`hrd_tools.tiled.walker` -- discovery + tiled-client registration
  helpers that mirror the source directory tree into nested containers
  and expose each v3 dataset as a bluesky-shaped run
  (``primary``/``baseline``).
- :mod:`hrd_tools.tiled.cli` -- ``python -m hrd_tools.tiled.cli`` entry
  point with ``register`` and ``init-config`` subcommands.

The cache layout is deliberately compatible with tiled's built-in
:class:`tiled.adapters.sparse_blocks_parquet.SparseBlocksParquetAdapter`
and :class:`tiled.adapters.parquet.ParquetDatasetAdapter`, so no custom
adapter code is required on the server side.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class V3Layout:
    """
    Format-level invariants for the v3 tiled cache layout.

    Every field here is a *naming or wire-format* constant that cannot
    be discovered by inspecting a dataset.  Values that the data
    determines (detector count, frame count, pixel grid, dtype) are
    carried through :class:`hrd_tools.tiled.rewrite.RewriteResult`.

    Attributes
    ----------
    manifest_name
        Bookkeeping file recorded next to the cache so re-running
        registration without changes is a no-op.
    detector_block_pattern
        Format string for per-detector per-block parquet file names.
        ``detector`` is 1-indexed and zero-padded to width 2.
        ``block`` is 0-indexed.
    scalars_name
        Name of the scalars table in the cache.
    baseline_name
        Name of the baseline table in the cache.
    images_basename, scalars_basename
        Source-side basenames for the images and scalars series.
    metadata_json_name
        Source-side metadata JSON file name.
    parquet_table_mimetype
        Mimetype for table parquet assets; mapped to
        :class:`tiled.adapters.parquet.ParquetDatasetAdapter` by
        default in tiled >= 0.2.9.
    parquet_sparse_mimetype
        Mimetype for sparse-block parquet assets; mapped to
        :class:`tiled.adapters.sparse_blocks_parquet.SparseBlocksParquetAdapter`
        by default in tiled >= 0.2.9.
    format_version
        Marker placed in run metadata.
    default_frames_per_chunk
        Default frame-axis chunk size for the cache.
    images_shape_metadata_key
        Schema-metadata key in source images parquet files containing
        the full ``[n_detectors, n_frames, n_rows, n_cols]`` shape.
    scalars_nominal_bin_metadata_key
        Schema-metadata key in source scalars parquet files containing
        the nominal 2theta bin size.
    """

    manifest_name: str = "cache_manifest.json"
    detector_block_pattern: str = "det_{detector:02d}_block-{block:d}.parquet"
    scalars_name: str = "scalars.parquet"
    baseline_name: str = "baseline.parquet"
    images_basename: str = "images"
    scalars_basename: str = "scalars"
    metadata_json_name: str = "metadata.json"
    parquet_table_mimetype: str = "application/x-parquet"
    parquet_sparse_mimetype: str = "application/x-parquet;structure=sparse"
    format_version: int = 3
    default_frames_per_chunk: int = 500
    images_shape_metadata_key: bytes = b"shape"
    scalars_nominal_bin_metadata_key: bytes = b"nominal_bin"


#: Singleton instance carrying the v3 format invariants.
LAYOUT: V3Layout = V3Layout()

__all__ = ["LAYOUT", "V3Layout"]
