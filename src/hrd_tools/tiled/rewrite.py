"""
Streaming rewrite of multihead v3 directories into a tiled-friendly cache.

The source v3 layout (see :class:`multihead.file_io.HRDRawV3`) stores
*all* detectors and *all* frames in a single ``images.parquet`` file (or
a sequence of frame-axis chunks ``images_000.parquet``, ...).  Each row
has columns ``[detector, frame, row, col, data]``.

Tiled's :class:`~tiled.adapters.sparse_blocks_parquet.SparseBlocksParquetAdapter`
expects **one parquet file per chunk** of a single sparse array, with
block-local coordinate columns followed by a final ``data`` column.  To
keep the runtime adapter trivial we therefore rewrite the source into
``n_detectors`` sibling sparse arrays, each chunked along the frame axis
with even-sized blocks (last block possibly shorter).  The number of
detectors and the pixel grid are *discovered* from the source schema
metadata rather than baked in.

The rewrite is streamed: we open a :class:`pyarrow.parquet.ParquetFile`
for each source file and iterate its row batches.  Each batch is
filtered by detector and partitioned into per-block sub-tables.  Output
``ParquetWriter`` instances are opened lazily on first write to a given
``(detector, block)`` target and closed when the rewrite finishes.

A small ``cache_manifest.json`` next to the rewritten files records the
``(path, size, mtime_ns)`` fingerprint of every source file consumed and
the rewrite parameters.  If a subsequent invocation sees a manifest that
matches the current source files exactly, the rewrite is skipped.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hrd_tools.tiled import LAYOUT

__all__ = [
    "FilesetFingerprint",
    "RewriteResult",
    "discover_image_files",
    "discover_scalar_files",
    "is_v3_directory",
    "load_manifest",
    "rewrite_v3_directory",
]


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


_NUMBERED_RE = re.compile(r"^(?P<base>[A-Za-z]+)_(?P<num>\d+)\.parquet$")


def _sorted_parquet_series(directory: Path, base: str) -> list[Path]:
    """
    Return ``[base.parquet]`` if it exists, else the sorted series
    ``base_NN.parquet`` (with no missing indices).

    Returns an empty list when no matching files are found.
    """
    single = directory / f"{base}.parquet"
    if single.exists():
        return [single]

    pairs: list[tuple[int, Path]] = []
    for p in directory.iterdir():
        m = _NUMBERED_RE.match(p.name)
        if m is None or m.group("base") != base:
            continue
        pairs.append((int(m.group("num")), p))

    if not pairs:
        return []

    pairs.sort(key=lambda kv: kv[0])
    nums = [n for n, _ in pairs]
    if nums != list(range(len(nums))):
        raise ValueError(
            f"Non-contiguous series for '{base}' in {directory}: found {nums}"
        )
    return [p for _, p in pairs]


def discover_image_files(directory: Path) -> list[Path]:
    """
    Return the ordered list of ``images*.parquet`` files in *directory*.
    """
    return _sorted_parquet_series(directory, LAYOUT.images_basename)


def discover_scalar_files(directory: Path) -> list[Path]:
    """
    Return the ordered list of ``scalars*.parquet`` files in *directory*.
    """
    return _sorted_parquet_series(directory, LAYOUT.scalars_basename)


def is_v3_directory(directory: Path) -> bool:
    """
    Cheap check: a v3 dataset directory has at least one images and one
    scalars parquet file.
    """
    if not directory.is_dir():
        return False
    return bool(discover_image_files(directory)) and bool(
        discover_scalar_files(directory)
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilesetFingerprint:
    """Identity record for a single source file."""

    path: str
    size: int
    mtime_ns: int

    @classmethod
    def of(cls, path: Path) -> FilesetFingerprint:
        st = path.stat()
        return cls(path=str(path.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns)

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FilesetFingerprint:
        return cls(
            path=data["path"], size=int(data["size"]), mtime_ns=int(data["mtime_ns"])
        )


@dataclass
class RewriteResult:
    """
    Outcome of :func:`rewrite_v3_directory`.

    Attributes
    ----------
    cache_dir
        The output directory containing the rewritten cache.
    skipped
        ``True`` when the manifest indicated nothing needed to be done.
    shape
        Full ``(n_detectors, n_frames, n_rows, n_cols)`` shape of the
        rewritten dataset.  This is the *discovered* shape -- consumers
        should use it instead of hardcoded constants.
    chunks
        Sizes of the frame-axis blocks (sums to ``shape[1]``).
    detector_block_paths
        Mapping ``{detector_1based: [block_path, ...]}``.
    scalars_path
        Path to the cached scalars parquet.
    baseline_path
        Path to the cached baseline parquet, or ``None`` if absent.
    metadata_json_path
        Path to the source ``metadata.json`` if present (not copied to
        cache; loaded directly for the run-level metadata).
    data_dtype
        Numpy dtype of the sparse ``data`` column.
    """

    cache_dir: Path
    skipped: bool
    shape: tuple[int, int, int, int]
    chunks: tuple[int, ...]
    detector_block_paths: dict[int, list[Path]]
    scalars_path: Path
    baseline_path: Path | None
    metadata_json_path: Path | None
    data_dtype: np.dtype = field(default_factory=lambda: np.dtype("uint32"))

    @property
    def n_detectors(self) -> int:
        return self.shape[0]

    @property
    def n_frames(self) -> int:
        return self.shape[1]

    @property
    def n_rows(self) -> int:
        return self.shape[2]

    @property
    def n_cols(self) -> int:
        return self.shape[3]


def load_manifest(cache_dir: Path) -> dict[str, Any] | None:
    """Return the manifest dict, or ``None`` if no manifest exists."""
    mpath = cache_dir / LAYOUT.manifest_name
    if not mpath.exists():
        return None
    with mpath.open("r") as fin:
        return json.load(fin)


def _manifest_matches(
    manifest: dict[str, Any],
    *,
    sources: list[Path],
    frames_per_chunk: int,
) -> bool:
    """Check whether *manifest* still describes the current source state."""
    if int(manifest.get("frames_per_chunk", -1)) != frames_per_chunk:
        return False
    expected = {str(p.resolve()): FilesetFingerprint.of(p) for p in sources}
    recorded_raw = manifest.get("sources", [])
    recorded = {rec["path"]: FilesetFingerprint.from_json(rec) for rec in recorded_raw}
    if set(expected) != set(recorded):
        return False
    return all(recorded[key] == fp for key, fp in expected.items())


# ---------------------------------------------------------------------------
# Streaming rewrite
# ---------------------------------------------------------------------------


def _read_shape(parquet_path: Path) -> tuple[int, int, int, int]:
    """
    Extract the ``shape`` schema metadata from a v3 images parquet file.

    The shape is ``(n_detectors, n_frames, n_rows, n_cols)`` for the
    portion of the dataset stored in that file.
    """
    md = pq.read_schema(parquet_path).metadata or {}
    raw = md.get(LAYOUT.images_shape_metadata_key)
    if raw is None:
        raise ValueError(f"Missing 'shape' metadata on {parquet_path}")
    shape = tuple(json.loads(raw))
    if len(shape) != 4:
        raise ValueError(f"Unexpected shape {shape} on {parquet_path}")
    return shape  # type: ignore[return-value]


def _block_sizes(n_frames: int, frames_per_chunk: int) -> tuple[int, ...]:
    """Even-sized blocks along the frame axis with a possibly shorter tail."""
    if n_frames <= 0:
        return ()
    full = n_frames // frames_per_chunk
    rem = n_frames - full * frames_per_chunk
    sizes = [frames_per_chunk] * full
    if rem:
        sizes.append(rem)
    return tuple(sizes)


_PARQUET_WRITE_KWARGS: dict[str, Any] = {
    "compression": "zstd",
    "use_dictionary": True,
}


def _symlink_or_copy(src: Path, dst: Path) -> None:
    """Symlink ``src`` to ``dst``, falling back to copying on failure."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _concatenate_scalars(scalar_paths: list[Path], dst: Path) -> None:
    """
    Concatenate (or symlink) the source scalars parquet files into *dst*.

    Pyarrow's :func:`pyarrow.concat_tables` is happy to concatenate
    tables with identical schemas.  When the source is a single file we
    just symlink/copy.  When there are multiple, we read+concat+write
    once -- this preserves the ``nominal_bin`` schema metadata from the
    first file (which all parts should agree on).
    """
    if len(scalar_paths) == 1:
        _symlink_or_copy(scalar_paths[0], dst)
        return
    tables = [pq.read_table(p) for p in scalar_paths]
    combined = pa.concat_tables(tables, promote_options="default")
    # Preserve schema metadata from the first table.
    combined = combined.replace_schema_metadata(tables[0].schema.metadata)
    pq.write_table(combined, dst, **_PARQUET_WRITE_KWARGS)


def rewrite_v3_directory(
    source_dir: Path,
    cache_dir: Path,
    *,
    frames_per_chunk: int = LAYOUT.default_frames_per_chunk,
    batch_size: int = 100_000,
) -> RewriteResult:
    """
    Rewrite a single v3 directory into the tiled-friendly cache layout.

    Parameters
    ----------
    source_dir
        Path to the v3 dataset directory (must satisfy
        :func:`is_v3_directory`).
    cache_dir
        Output directory.  Created if needed.  Must be a separate path
        from *source_dir*.
    frames_per_chunk
        Number of frames per output block.  Pixel dimensions are not
        chunked.
    batch_size
        Row-batch size when iterating the source parquet files.  The
        peak memory cost of the rewrite is roughly
        ``batch_size * row-width`` plus one open ``ParquetWriter`` per
        ``(detector, block)`` that has seen data so far.

    Returns
    -------
    RewriteResult
        Describes the contents of the cache and whether the rewrite was
        skipped because the manifest matched the source state.

    Notes
    -----
    The function is idempotent: if a manifest already in *cache_dir*
    matches the current source files (same ``path``/``size``/``mtime``
    and ``frames_per_chunk``), no parquet writes occur and the existing
    cache is reused.
    """
    source_dir = Path(source_dir)
    cache_dir = Path(cache_dir)

    if source_dir.resolve() == cache_dir.resolve():
        raise ValueError("cache_dir must differ from source_dir")

    if not is_v3_directory(source_dir):
        raise ValueError(f"{source_dir} is not a v3 directory")

    image_paths = discover_image_files(source_dir)
    scalar_paths = discover_scalar_files(source_dir)
    baseline_src = source_dir / LAYOUT.baseline_name
    metadata_src = source_dir / LAYOUT.metadata_json_name

    # Discover and cross-check the global shape.  The frame axis (dim 1)
    # is summed across files; dims 0, 2, and 3 must agree.
    file_shapes = [_read_shape(p) for p in image_paths]
    static_dims = {(s[0], s[2], s[3]) for s in file_shapes}
    if len(static_dims) != 1:
        raise ValueError(
            "images parquet files disagree on (n_detectors, n_rows, n_cols);"
            f" saw {static_dims} in {source_dir}"
        )
    n_detectors, n_rows, n_cols = next(iter(static_dims))
    if n_detectors <= 0:
        raise ValueError(f"non-positive n_detectors {n_detectors} in {source_dir}")

    file_frame_counts = [s[1] for s in file_shapes]
    n_frames = sum(file_frame_counts)
    global_shape: tuple[int, int, int, int] = (
        n_detectors,
        n_frames,
        n_rows,
        n_cols,
    )
    chunks = _block_sizes(n_frames, frames_per_chunk)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- Incremental skip ------------------------------------------------
    all_sources = [*image_paths, *scalar_paths]
    if baseline_src.exists():
        all_sources.append(baseline_src)
    if metadata_src.exists():
        all_sources.append(metadata_src)

    manifest = load_manifest(cache_dir)
    if manifest is not None and _manifest_matches(
        manifest, sources=all_sources, frames_per_chunk=frames_per_chunk
    ):
        scalars_dst = cache_dir / LAYOUT.scalars_name
        baseline_dst = cache_dir / LAYOUT.baseline_name
        # Prefer the manifest's recorded shape (it captures what was
        # actually written); fall back to the freshly discovered one.
        recorded_shape = manifest.get("shape")
        if recorded_shape is not None and len(recorded_shape) == 4:
            cached_shape: tuple[int, int, int, int] = tuple(
                int(x) for x in recorded_shape
            )  # type: ignore[assignment]
        else:
            cached_shape = global_shape
        det_paths = {
            d: [
                cache_dir / LAYOUT.detector_block_pattern.format(detector=d, block=k)
                for k in range(len(chunks))
            ]
            for d in range(1, cached_shape[0] + 1)
        }
        return RewriteResult(
            cache_dir=cache_dir,
            skipped=True,
            shape=cached_shape,
            chunks=chunks,
            detector_block_paths=det_paths,
            scalars_path=scalars_dst,
            baseline_path=baseline_dst if baseline_dst.exists() else None,
            metadata_json_path=metadata_src if metadata_src.exists() else None,
            data_dtype=np.dtype(manifest.get("data_dtype", "uint32")),
        )

    # ----- Pre-clean previous block parquet files --------------------------
    # We do not blow away the whole directory in case there are
    # unrelated files (we never put any, but be defensive).
    for stale in cache_dir.glob("det_*_block-*.parquet"):
        stale.unlink()

    # ----- Streaming rewrite ----------------------------------------------
    # writers: (detector_1based, block_idx) -> ParquetWriter
    writers: dict[tuple[int, int], pq.ParquetWriter] = {}
    detector_block_paths: dict[int, list[Path]] = {
        d: [] for d in range(1, n_detectors + 1)
    }

    # Pre-compute the schema for output files.  The data type comes from
    # the first source file's ``data`` column.
    first_schema = pq.read_schema(image_paths[0])
    for required in ("detector", "frame", "row", "col", "data"):
        if first_schema.get_field_index(required) < 0:
            raise ValueError(
                f"images parquet {image_paths[0]} missing column {required!r}"
            )
    data_dtype: pa.DataType = first_schema.field("data").type
    frame_dtype: pa.DataType = first_schema.field("frame").type
    row_dtype: pa.DataType = first_schema.field("row").type
    col_dtype: pa.DataType = first_schema.field("col").type

    out_schema = pa.schema(
        [
            ("frame", frame_dtype),
            ("row", row_dtype),
            ("col", col_dtype),
            ("data", data_dtype),
        ]
    )

    def _open_writer(det: int, block: int) -> pq.ParquetWriter:
        key = (det, block)
        if key in writers:
            return writers[key]
        path = cache_dir / LAYOUT.detector_block_pattern.format(
            detector=det, block=block
        )
        detector_block_paths[det].append(path)
        writer = pq.ParquetWriter(path, out_schema, **_PARQUET_WRITE_KWARGS)
        writers[key] = writer
        return writer

    try:
        global_frame_offset = 0
        for src_path, (_, file_n_frames, _, _) in zip(
            image_paths, file_shapes, strict=True
        ):
            pf = pq.ParquetFile(src_path)
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=["detector", "frame", "row", "col", "data"],
            ):
                # Shift frame to global coordinates.
                global_frames = pa.compute.add(
                    pa.compute.cast(batch.column("frame"), pa.int64()),
                    pa.scalar(global_frame_offset, type=pa.int64()),
                )
                # Block index along the frame axis.
                block_idx = pa.compute.divide(
                    global_frames, pa.scalar(frames_per_chunk, type=pa.int64())
                )
                # Block-local frame offset.
                local_frame = pa.compute.subtract(
                    global_frames,
                    pa.compute.multiply(
                        block_idx, pa.scalar(frames_per_chunk, type=pa.int64())
                    ),
                )
                # Cast local frame back to the original frame dtype so
                # the output schema stays stable.
                local_frame = pa.compute.cast(local_frame, frame_dtype)

                detector_arr = batch.column("detector")
                # We need to partition (detector, block_idx).  Pyarrow
                # has no direct groupby->iter, so we fall back to numpy.
                det_np = detector_arr.to_numpy(zero_copy_only=False).astype(
                    np.int64, copy=False
                )
                block_np = block_idx.to_numpy(zero_copy_only=False).astype(
                    np.int64, copy=False
                )

                # combined key = det * (n_blocks + 1) + block
                n_blocks = len(chunks) if chunks else 1
                combined = det_np * (n_blocks + 1) + block_np
                order = np.argsort(combined, kind="stable")
                if order.size == 0:
                    continue
                sorted_combined = combined[order]
                # Find run boundaries.
                breaks = np.flatnonzero(np.diff(sorted_combined)) + 1
                segment_starts = np.concatenate(([0], breaks))
                segment_ends = np.concatenate((breaks, [sorted_combined.size]))

                # Apply the sort to every column once.
                indices = pa.array(order)
                sorted_local_frame = local_frame.take(indices)
                sorted_row = batch.column("row").take(indices)
                sorted_col = batch.column("col").take(indices)
                sorted_data = batch.column("data").take(indices)
                sorted_det = det_np[order]
                sorted_block = block_np[order]

                for s, e in zip(segment_starts, segment_ends, strict=True):
                    det_0 = int(sorted_det[s])
                    blk = int(sorted_block[s])
                    if not (0 <= det_0 < n_detectors):
                        raise ValueError(
                            f"detector value {det_0} out of range [0,"
                            f" {n_detectors}) in {src_path}"
                        )
                    if not (0 <= blk < len(chunks)):
                        raise ValueError(
                            f"block index {blk} out of range [0, {len(chunks)})"
                            f" derived from global frame in {src_path}"
                        )
                    sub = pa.table(
                        {
                            "frame": sorted_local_frame.slice(s, e - s),
                            "row": sorted_row.slice(s, e - s),
                            "col": sorted_col.slice(s, e - s),
                            "data": sorted_data.slice(s, e - s),
                        }
                    )
                    writer = _open_writer(det_0 + 1, blk)
                    writer.write_table(sub)

            global_frame_offset += file_n_frames
    finally:
        for writer in writers.values():
            writer.close()

    # Ensure every (det, block) target exists -- some detectors / blocks
    # may have had zero non-zero pixels in the source.  Empty parquet
    # files keep the tiled chunk grid contiguous.
    for det in range(1, n_detectors + 1):
        for blk in range(len(chunks)):
            path = cache_dir / LAYOUT.detector_block_pattern.format(
                detector=det, block=blk
            )
            if not path.exists():
                empty = pa.table(
                    {
                        "frame": pa.array([], type=frame_dtype),
                        "row": pa.array([], type=row_dtype),
                        "col": pa.array([], type=col_dtype),
                        "data": pa.array([], type=data_dtype),
                    }
                )
                pq.write_table(empty, path, **_PARQUET_WRITE_KWARGS)
                detector_block_paths[det].append(path)
        detector_block_paths[det].sort(key=lambda p: int(p.stem.rsplit("-", 1)[1]))

    # ----- Sidecars (scalars, baseline) -----------------------------------
    scalars_dst = cache_dir / LAYOUT.scalars_name
    _concatenate_scalars(scalar_paths, scalars_dst)

    baseline_dst: Path | None = None
    if baseline_src.exists():
        baseline_dst = cache_dir / LAYOUT.baseline_name
        _symlink_or_copy(baseline_src, baseline_dst)

    # ----- Manifest --------------------------------------------------------
    manifest_payload = {
        "tool": "hrd_tools.tiled.rewrite",
        "format_version": LAYOUT.format_version,
        "frames_per_chunk": frames_per_chunk,
        "shape": list(global_shape),
        "chunks": list(chunks),
        "data_dtype": str(np.dtype(data_dtype.to_pandas_dtype())),
        "sources": [FilesetFingerprint.of(p).to_json() for p in all_sources],
    }
    with (cache_dir / LAYOUT.manifest_name).open("w") as fout:
        json.dump(manifest_payload, fout, indent=2, sort_keys=True)

    return RewriteResult(
        cache_dir=cache_dir,
        skipped=False,
        shape=global_shape,
        chunks=chunks,
        detector_block_paths=detector_block_paths,
        scalars_path=scalars_dst,
        baseline_path=baseline_dst,
        metadata_json_path=metadata_src if metadata_src.exists() else None,
        data_dtype=np.dtype(data_dtype.to_pandas_dtype()),
    )
