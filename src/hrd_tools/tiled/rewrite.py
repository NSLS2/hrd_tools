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
import pyarrow.compute as pc
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
    for stale in cache_dir.glob("det_*_block-*.part-*.parquet"):
        stale.unlink()

    # ----- Streaming rewrite ----------------------------------------------
    # Within a single v3 images parquet file, rows are emitted in
    # ``(detector, frame)`` lex order: every detector's full frame
    # range appears contiguously and frames are non-decreasing within
    # a detector.  Because frame-axis blocks are contiguous ranges of
    # frames, the sequence of ``(detector, block)`` cells observed
    # *within a single file* is also non-decreasing in lex order, and
    # once we advance from one cell to the next we never return to the
    # previous one (within that file).
    #
    # Across files in a numbered series the data is concatenated along
    # the frame axis, so the second file starts at frame
    # ``sum(prior_file_frames)`` and again iterates detector-major.
    # That means a ``(det, block)`` cell can legitimately receive
    # writes from more than one source file -- but for a given cell
    # those writes are still globally in frame order.
    #
    # The rewrite therefore:
    #
    # * Keeps exactly **one** open ``ParquetWriter`` at a time, for
    #   the active ``(det, block)`` cell within the current file.
    # * Closes that writer as soon as the cell advances within the
    #   file (no row from this file will return to it).
    # * Writes each per-file contribution to a ``det_NN_block-K.part-J.parquet``
    #   file, where ``J`` is the source file index.
    # * After streaming, consolidates the parts of each cell into the
    #   canonical ``det_NN_block-K.parquet`` -- with one reader and
    #   one writer open at a time, so peak fd usage during the whole
    #   pipeline is ~3.
    #
    # Within each file we enforce strict ``(det, block)`` monotonicity:
    # going backwards inside a single file is a hard error.
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

    detector_block_paths: dict[int, list[Path]] = {
        d: [] for d in range(1, n_detectors + 1)
    }
    # ``cell_parts[(det, block)]`` lists the per-file part paths for
    # that cell, in source-file order.  Most cells will have at most
    # one part; only cells that straddle a source-file boundary will
    # have multiple.
    cell_parts: dict[tuple[int, int], list[Path]] = {}

    def _block_path(det: int, block: int) -> Path:
        return cache_dir / LAYOUT.detector_block_pattern.format(
            detector=det, block=block
        )

    def _part_path(det: int, block: int, file_idx: int) -> Path:
        return cache_dir / f"det_{det:02d}_block-{block:d}.part-{file_idx:d}.parquet"

    # State for the currently open writer (within one source file).
    current_cell: tuple[int, int] | None = None
    current_writer: pq.ParquetWriter | None = None

    def _close_current() -> None:
        nonlocal current_writer, current_cell
        if current_writer is not None:
            current_writer.close()
            current_writer = None
        current_cell = None

    try:
        global_frame_offset = 0
        for file_idx, (src_path, (_, file_n_frames, _, _)) in enumerate(
            zip(image_paths, file_shapes, strict=True)
        ):
            # Each source file starts a fresh (det, block) sequence.
            # Within the file, cells are visited in lex order; across
            # files we may revisit cells (that straddle the file
            # boundary), so each file writes to its own part file.
            _close_current()

            pf = pq.ParquetFile(src_path)
            for batch in pf.iter_batches(
                batch_size=batch_size,
                columns=["detector", "frame", "row", "col", "data"],
            ):
                if batch.num_rows == 0:
                    continue

                # Shift frame to global coordinates.
                global_frames = pc.add(
                    pc.cast(batch.column("frame"), pa.int64()),
                    pa.scalar(global_frame_offset, type=pa.int64()),
                )
                # Block index along the frame axis.
                block_idx = pc.divide(
                    global_frames, pa.scalar(frames_per_chunk, type=pa.int64())
                )
                # Block-local frame offset.
                local_frame = pc.subtract(
                    global_frames,
                    pc.multiply(
                        block_idx, pa.scalar(frames_per_chunk, type=pa.int64())
                    ),
                )
                # Cast local frame back to the original frame dtype so
                # the output schema stays stable.
                local_frame = pc.cast(local_frame, frame_dtype)

                block_np = block_idx.to_numpy(zero_copy_only=False).astype(
                    np.int64, copy=False
                )
                det_np = (
                    batch.column("detector")
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int64, copy=False)
                )

                # Validate ranges up-front.
                dmin = int(det_np.min())
                dmax = int(det_np.max())
                if dmin < 0 or dmax >= n_detectors:
                    raise ValueError(
                        f"Detector value out of range [0, {n_detectors})"
                        f" in {src_path}: saw [{dmin}, {dmax}]"
                    )
                bmin = int(block_np.min())
                bmax = int(block_np.max())
                if bmin < 0 or bmax >= len(chunks):
                    raise ValueError(
                        f"Batch block range [{bmin}, {bmax}] out of"
                        f" bounds [0, {len(chunks)}) in {src_path}"
                    )

                # Compute the sequential cell key = det * n_blocks + block.
                # Within a single source file the sequence of cell
                # keys must be non-decreasing.
                n_blocks = max(len(chunks), 1)
                cell_key = det_np * n_blocks + block_np
                if batch.num_rows > 1 and not np.all(np.diff(cell_key) >= 0):
                    bad_idx = int(np.argmax(np.diff(cell_key) < 0))
                    raise ValueError(
                        f"Input is not sequential in (detector, frame) lex"
                        f" order within {src_path}: row {bad_idx} is"
                        f" (det={int(det_np[bad_idx])},"
                        f" block={int(block_np[bad_idx])}); row"
                        f" {bad_idx + 1} is"
                        f" (det={int(det_np[bad_idx + 1])},"
                        f" block={int(block_np[bad_idx + 1])})."
                    )
                if current_cell is not None:
                    first_key = int(cell_key[0])
                    active_key = (current_cell[0] - 1) * n_blocks + current_cell[1]
                    if first_key < active_key:
                        raise ValueError(
                            f"Input is not sequential in (detector, frame)"
                            f" lex order within {src_path}: batch starts"
                            f" at (det={int(det_np[0])},"
                            f" block={int(block_np[0])}) but the active"
                            f" cell is {current_cell}."
                        )

                # Split the batch into contiguous same-cell runs.
                cell_breaks = np.flatnonzero(np.diff(cell_key)) + 1
                seg_starts = np.concatenate(([0], cell_breaks))
                seg_ends = np.concatenate((cell_breaks, [cell_key.size]))

                for s, e in zip(seg_starts, seg_ends, strict=True):
                    det_0 = int(det_np[s])
                    blk = int(block_np[s])
                    cell = (det_0 + 1, blk)
                    if cell != current_cell:
                        # Advance to a new cell within this file.  The
                        # per-file lex-order invariant guarantees we
                        # will not return to the previous one before
                        # the file ends.
                        if current_writer is not None:
                            current_writer.close()
                            current_writer = None
                        path = _part_path(*cell, file_idx)
                        cell_parts.setdefault(cell, []).append(path)
                        current_writer = pq.ParquetWriter(
                            path, out_schema, **_PARQUET_WRITE_KWARGS
                        )
                        current_cell = cell
                    assert current_writer is not None
                    sub = pa.table(
                        {
                            "frame": local_frame.slice(s, e - s),
                            "row": batch.column("row").slice(s, e - s),
                            "col": batch.column("col").slice(s, e - s),
                            "data": batch.column("data").slice(s, e - s),
                        }
                    )
                    current_writer.write_table(sub)

            global_frame_offset += file_n_frames
    finally:
        _close_current()

    # ----- Consolidate parts into the canonical block parquet files -------
    # For each cell, merge its (typically one, occasionally more) part
    # files into the single ``det_NN_block-K.parquet`` output and
    # unlink the parts.  At most one ParquetWriter and one ParquetFile
    # are open at a time; finalization fd usage is bounded.
    empty_template = pa.table(
        {
            "frame": pa.array([], type=frame_dtype),
            "row": pa.array([], type=row_dtype),
            "col": pa.array([], type=col_dtype),
            "data": pa.array([], type=data_dtype),
        }
    )
    for det in range(1, n_detectors + 1):
        for blk in range(len(chunks)):
            final_path = _block_path(det, blk)
            parts = cell_parts.get((det, blk), [])
            if not parts:
                # No data: write an empty parquet so the tiled chunk
                # grid stays contiguous.
                pq.write_table(empty_template, final_path, **_PARQUET_WRITE_KWARGS)
            elif len(parts) == 1:
                # Common fast path.
                parts[0].replace(final_path)
            else:
                writer = pq.ParquetWriter(
                    final_path, out_schema, **_PARQUET_WRITE_KWARGS
                )
                try:
                    for part_path in parts:
                        pf = pq.ParquetFile(part_path)
                        for batch in pf.iter_batches(batch_size=batch_size):
                            writer.write_table(pa.Table.from_batches([batch]))
                finally:
                    writer.close()
                for part_path in parts:
                    if part_path.exists():
                        part_path.unlink()
            detector_block_paths[det].append(final_path)

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
