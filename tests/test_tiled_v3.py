"""
Tests for :mod:`hrd_tools.tiled`.

These exercise the streaming rewriter against a synthetic v3 directory,
then drive the registration helpers against an in-process tiled
catalog.  Everything is skipped when ``multihead`` or ``tiled`` is not
available.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("multihead")
pytest.importorskip("tiled")
pytest.importorskip("pyarrow.parquet")
pytest.importorskip("pyarrow")
pytest.importorskip("sparse")
pytest.importorskip("yaml")

import pyarrow as pa
import pyarrow.parquet as pq
import sparse
import yaml
from multihead.file_io import HRDRawV3

from hrd_tools.tiled import LAYOUT, cli
from hrd_tools.tiled.rewrite import (
    discover_image_files,
    discover_scalar_files,
    is_v3_directory,
    rewrite_v3_directory,
)
from hrd_tools.tiled.walker import (
    discover_v3_directories,
    register_root,
    register_run,
)

# Default fixture detector count chosen for test speed; specific tests
# parametrize over a small set of shapes to lock in shape independence.
DEFAULT_N_DETECTORS = 2
DEFAULT_PIXEL_SIZE = 256


# ---------------------------------------------------------------------------
# Synthetic data construction
# ---------------------------------------------------------------------------


def _build_sparse_array(
    *,
    n_frames: int,
    n_detectors: int = DEFAULT_N_DETECTORS,
    pixel_size: int = DEFAULT_PIXEL_SIZE,
    seed: int = 0,
    density: int = 80,
) -> sparse.COO:
    """
    Build a deterministic sparse COO of shape
    ``(n_detectors, n_frames, pixel_size, pixel_size)`` with ``density``
    non-zero entries per detector.
    """
    rng = np.random.default_rng(seed)
    total = n_detectors * density
    detector = np.repeat(np.arange(n_detectors, dtype=np.uint8), density)
    frame = rng.integers(0, n_frames, size=total).astype(np.uint32)
    row = rng.integers(0, pixel_size, size=total).astype(np.uint16)
    col = rng.integers(0, pixel_size, size=total).astype(np.uint16)
    data = rng.integers(1, 1000, size=total, dtype=np.uint32)
    return sparse.COO(
        coords=np.stack([detector, frame, row, col]),
        data=data,
        shape=(n_detectors, n_frames, pixel_size, pixel_size),
    ).asformat("coo")


def _write_images_parquet(arr: sparse.COO, dest: Path) -> None:
    """Write a v3-style images parquet file from a sparse COO."""
    table = pa.table(
        {
            "detector": pa.array(arr.coords[0], type=pa.uint8()),
            "frame": pa.array(arr.coords[1], type=pa.uint32()),
            "row": pa.array(arr.coords[2], type=pa.uint16()),
            "col": pa.array(arr.coords[3], type=pa.uint16()),
            "data": pa.array(arr.data, type=pa.uint32()),
        }
    )
    table = table.replace_schema_metadata(
        {b"shape": json.dumps(list(arr.shape)).encode()}
    )
    pq.write_table(table, dest, compression="zstd")


def _write_scalars_parquet(
    n_frames: int,
    dest: Path,
    *,
    nominal_bin: float = 0.001,
    include_extras: bool = False,
) -> None:
    tth = np.arange(n_frames, dtype=np.float64) * nominal_bin + 1.0
    mon = np.full(n_frames, 1_000_000.0, dtype=np.float64)
    cols = {"tth": pa.array(tth), "monitor": pa.array(mon)}
    if include_extras:
        cols["I0"] = pa.array(mon * 0.9)
    table = pa.table(cols)
    table = table.replace_schema_metadata({b"nominal_bin": str(nominal_bin).encode()})
    pq.write_table(table, dest)


def _write_baseline_parquet(dest: Path) -> None:
    table = pa.table(
        {
            "timestamp": pa.array(
                ["2026-01-01T00:00:00", "2026-01-01T00:10:00"], type=pa.string()
            ),
            "PV:A": pa.array([1.0, 1.1], type=pa.float64()),
            "PV:B": pa.array(["off", "on"], type=pa.string()),
        }
    )
    pq.write_table(table, dest)


def _write_metadata_json(dest: Path) -> dict:
    payload = {
        "scan_md": {"sample": "Si standard", "operator": "test"},
        "scan_config": {"NPTS": {"value": [7], "unit": "", "pv": "fake:NPTS"}},
        "staff_log": [{"timestamp": "now", "steps": []}],
    }
    with dest.open("w") as fout:
        json.dump(payload, fout)
    return payload


def _make_v3_dir(
    parent: Path,
    name: str,
    *,
    n_frames: int = 7,
    n_detectors: int = DEFAULT_N_DETECTORS,
    pixel_size: int = DEFAULT_PIXEL_SIZE,
    seed: int = 0,
    with_baseline: bool = True,
    with_metadata: bool = True,
) -> tuple[Path, sparse.COO, dict | None]:
    """Create a one-directory v3 dataset and return path + truth values."""
    arr = _build_sparse_array(
        n_frames=n_frames,
        n_detectors=n_detectors,
        pixel_size=pixel_size,
        seed=seed,
    )
    src = parent / name
    src.mkdir()
    _write_images_parquet(arr, src / "images.parquet")
    _write_scalars_parquet(n_frames, src / "scalars.parquet")
    if with_baseline:
        _write_baseline_parquet(src / "baseline.parquet")
    md: dict | None = None
    if with_metadata:
        md = _write_metadata_json(src / "metadata.json")
    return src, arr, md


@pytest.fixture
def synth_v3_dir(tmp_path: Path) -> tuple[Path, sparse.COO, dict]:
    """
    Default 2-detector v3 dataset.  Returns ``(path, sparse_array,
    metadata_payload)``.
    """
    src, arr, md = _make_v3_dir(tmp_path, "run_017")
    assert md is not None
    return src, arr, md


@pytest.fixture
def synth_v3_series_dir(tmp_path: Path) -> tuple[Path, sparse.COO]:
    """
    A v3 dataset split across two images files and two scalars files.
    """
    full_n = 9
    arr = _build_sparse_array(
        n_frames=full_n,
        n_detectors=DEFAULT_N_DETECTORS,
        pixel_size=DEFAULT_PIXEL_SIZE,
        seed=42,
        density=30,
    )
    src = tmp_path / "run_018_series"
    src.mkdir()

    # Split along the frame axis at 5 / 9.
    split = 5
    a_coords = arr.coords[:, arr.coords[1] < split]
    a_data = arr.data[arr.coords[1] < split]
    b_coords = arr.coords[:, arr.coords[1] >= split].copy()
    b_data = arr.data[arr.coords[1] >= split]
    b_coords[1] -= split  # local frame index inside the second file
    a = sparse.COO(
        coords=a_coords,
        data=a_data,
        shape=(DEFAULT_N_DETECTORS, split, DEFAULT_PIXEL_SIZE, DEFAULT_PIXEL_SIZE),
    ).asformat("coo")
    b = sparse.COO(
        coords=b_coords,
        data=b_data,
        shape=(
            DEFAULT_N_DETECTORS,
            full_n - split,
            DEFAULT_PIXEL_SIZE,
            DEFAULT_PIXEL_SIZE,
        ),
    ).asformat("coo")
    _write_images_parquet(a, src / "images_0.parquet")
    _write_images_parquet(b, src / "images_1.parquet")

    _write_scalars_parquet(split, src / "scalars_0.parquet")
    _write_scalars_parquet(
        full_n - split,
        src / "scalars_1.parquet",
    )
    return src, arr


# ---------------------------------------------------------------------------
# Discovery and is_v3_directory
# ---------------------------------------------------------------------------


def test_is_v3_directory_positive(synth_v3_dir: tuple[Path, sparse.COO, dict]):
    src, _, _ = synth_v3_dir
    assert is_v3_directory(src)
    assert discover_image_files(src) == [src / "images.parquet"]
    assert discover_scalar_files(src) == [src / "scalars.parquet"]


def test_is_v3_directory_series(synth_v3_series_dir: tuple[Path, sparse.COO]):
    src, _ = synth_v3_series_dir
    assert is_v3_directory(src)
    assert [p.name for p in discover_image_files(src)] == [
        "images_0.parquet",
        "images_1.parquet",
    ]
    assert [p.name for p in discover_scalar_files(src)] == [
        "scalars_0.parquet",
        "scalars_1.parquet",
    ]


def test_is_v3_directory_negative(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not is_v3_directory(empty)
    # File, not directory.
    f = tmp_path / "file"
    f.write_text("x")
    assert not is_v3_directory(f)


def test_discover_v3_directories_nested(
    tmp_path: Path, synth_v3_dir: tuple[Path, sparse.COO, dict]
):
    src, _, _ = synth_v3_dir
    # Move src into a nested layout.
    root = tmp_path / "root"
    nested = root / "2026-01" / "session_A"
    nested.mkdir(parents=True)
    target = nested / src.name
    src.rename(target)
    found = list(discover_v3_directories(root))
    assert found == [target]


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------


def _read_block(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (frame, row, col, data) numpy arrays from a cache block."""
    t = pq.read_table(path)
    return (
        t.column("frame").to_numpy(),
        t.column("row").to_numpy(),
        t.column("col").to_numpy(),
        t.column("data").to_numpy(),
    )


def _reconstruct_detector(
    block_paths: list[Path],
    chunks: tuple[int, ...],
    n_frames: int,
    pixel_size: int,
) -> sparse.COO:
    """Reassemble a per-detector sparse array from cache block files."""
    if len(block_paths) != len(chunks):
        raise AssertionError(
            f"block count {len(block_paths)} != chunk count {len(chunks)}"
        )
    all_frame: list[np.ndarray] = []
    all_row: list[np.ndarray] = []
    all_col: list[np.ndarray] = []
    all_data: list[np.ndarray] = []
    offset = 0
    for path, csize in zip(block_paths, chunks, strict=True):
        f, r, c, d = _read_block(path)
        all_frame.append(f.astype(np.int64) + offset)
        all_row.append(r.astype(np.int64))
        all_col.append(c.astype(np.int64))
        all_data.append(d)
        offset += csize
    return sparse.COO(
        coords=np.stack(
            [
                np.concatenate(all_frame),
                np.concatenate(all_row),
                np.concatenate(all_col),
            ]
        ),
        data=np.concatenate(all_data),
        shape=(n_frames, pixel_size, pixel_size),
    ).asformat("coo")


@pytest.mark.parametrize(
    ("n_detectors", "pixel_size"),
    [
        (2, 256),
        (1, 128),
        (3, 192),
    ],
)
def test_rewrite_single_file_round_trip(
    tmp_path: Path, n_detectors: int, pixel_size: int
):
    src, arr, _ = _make_v3_dir(
        tmp_path,
        "run_017",
        n_frames=7,
        n_detectors=n_detectors,
        pixel_size=pixel_size,
    )
    cache = tmp_path / "cache"
    res = rewrite_v3_directory(src, cache, frames_per_chunk=3)
    assert not res.skipped
    assert res.shape == (n_detectors, 7, pixel_size, pixel_size)
    assert res.n_frames == 7
    assert res.chunks == (3, 3, 1)
    assert (cache / LAYOUT.scalars_name).exists()
    assert (cache / LAYOUT.baseline_name).exists()
    assert (cache / LAYOUT.manifest_name).exists()
    # Every (det, block) parquet exists.
    for det in range(1, n_detectors + 1):
        for blk in range(3):
            path = cache / LAYOUT.detector_block_pattern.format(detector=det, block=blk)
            assert path.exists(), f"missing {path}"

    # Reconstruct and compare to the original sparse array (axis-0 slice).
    dense = arr.todense()
    for det in range(1, n_detectors + 1):
        expected = dense[det - 1]
        got = _reconstruct_detector(
            res.detector_block_paths[det], res.chunks, res.n_frames, pixel_size
        )
        assert got.shape == expected.shape
        assert np.array_equal(got.todense(), expected)


def test_rewrite_matches_hrdrawv3_oracle(
    tmp_path: Path, synth_v3_dir: tuple[Path, sparse.COO, dict]
):
    """Cross-check the rewrite against the upstream HRDRawV3 reader."""
    src, _, _ = synth_v3_dir
    cache = tmp_path / "cache"
    res = rewrite_v3_directory(src, cache, frames_per_chunk=3)
    reader = HRDRawV3(src)
    for det in range(1, res.n_detectors + 1):
        expected = reader.get_detector(det)
        got = _reconstruct_detector(
            res.detector_block_paths[det],
            res.chunks,
            res.n_frames,
            res.n_rows,
        )
        assert got.shape == expected.shape
        assert np.array_equal(got.todense(), expected.todense())


def test_rewrite_local_frame_offsets(
    tmp_path: Path, synth_v3_dir: tuple[Path, sparse.COO, dict]
):
    src, _, _ = synth_v3_dir
    cache = tmp_path / "cache"
    res = rewrite_v3_directory(src, cache, frames_per_chunk=3)
    # Block 0 has frames 0..2; block 1 has 3..5; block 2 has frame 6 only.
    # Local frame values in each block must be in [0, chunk_size).
    for det in range(1, res.n_detectors + 1):
        for blk, csize in enumerate(res.chunks):
            f, _, _, _ = _read_block(res.detector_block_paths[det][blk])
            if len(f) == 0:
                continue
            assert f.min() >= 0
            assert f.max() < csize, (det, blk, f.max(), csize)


def test_rewrite_series_round_trip(
    tmp_path: Path, synth_v3_series_dir: tuple[Path, sparse.COO]
):
    src, _ = synth_v3_series_dir
    cache = tmp_path / "cache"
    res = rewrite_v3_directory(src, cache, frames_per_chunk=4)
    assert res.shape == (
        DEFAULT_N_DETECTORS,
        9,
        DEFAULT_PIXEL_SIZE,
        DEFAULT_PIXEL_SIZE,
    )
    assert res.chunks == (4, 4, 1)

    reader = HRDRawV3(src)
    for det in range(1, DEFAULT_N_DETECTORS + 1):
        expected = reader.get_detector(det)
        got = _reconstruct_detector(
            res.detector_block_paths[det],
            res.chunks,
            res.n_frames,
            DEFAULT_PIXEL_SIZE,
        )
        assert np.array_equal(got.todense(), expected.todense())


def test_rewrite_streaming_small_batches(tmp_path: Path):
    """
    A small ``batch_size`` forces many iter_batches loops with batches
    that may straddle frame-chunk boundaries.  The sequential rewriter
    must still close per-block writers as soon as the block advances,
    and produce a faithful cache.
    """
    src, arr, _ = _make_v3_dir(
        tmp_path,
        "run_stream",
        n_frames=12,
        n_detectors=3,
        pixel_size=64,
        seed=11,
    )
    cache = tmp_path / "cache_stream"
    res = rewrite_v3_directory(src, cache, frames_per_chunk=2, batch_size=8)
    assert not res.skipped
    assert res.shape == (3, 12, 64, 64)
    assert res.chunks == (2, 2, 2, 2, 2, 2)

    # Every (det, block) canonical file exists.
    for det in range(1, 4):
        for blk in range(len(res.chunks)):
            path = cache / LAYOUT.detector_block_pattern.format(detector=det, block=blk)
            assert path.exists(), f"missing {path}"

    # Round-trip the data through reconstruction.
    dense = arr.todense()
    for det in range(1, 4):
        got = _reconstruct_detector(
            res.detector_block_paths[det], res.chunks, res.n_frames, 64
        )
        assert np.array_equal(got.todense(), dense[det - 1])


def _write_raw_images_parquet(
    detector: np.ndarray,
    frame: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int, int, int],
    dest: Path,
) -> None:
    """Write an images parquet directly from raw (possibly unsorted) arrays."""
    table = pa.table(
        {
            "detector": pa.array(detector, type=pa.uint8()),
            "frame": pa.array(frame, type=pa.uint32()),
            "row": pa.array(row, type=pa.uint16()),
            "col": pa.array(col, type=pa.uint16()),
            "data": pa.array(data, type=pa.uint32()),
        }
    )
    table = table.replace_schema_metadata({b"shape": json.dumps(list(shape)).encode()})
    pq.write_table(table, dest, compression="zstd")


def test_rewrite_rejects_frames_out_of_order_within_detector(tmp_path: Path):
    """
    Within a single detector run, frames must be non-decreasing.  A
    scrambled-frame images parquet must be rejected rather than
    silently corrupted.
    """
    n_frames = 10
    n_detectors = 2
    pixel_size = 32
    # Construct two detector runs (det 0 then det 1).  Within det 0,
    # insert a backward frame jump that crosses a block boundary at
    # frames_per_chunk=3 (so blocks change too).
    detector = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.uint8)
    frame = np.array([0, 1, 6, 2, 0, 4, 8], dtype=np.uint32)
    row = np.zeros(7, dtype=np.uint16)
    col = np.zeros(7, dtype=np.uint16)
    data = np.arange(1, 8, dtype=np.uint32)

    src = tmp_path / "run_scrambled_det"
    src.mkdir()
    _write_raw_images_parquet(
        detector,
        frame,
        row,
        col,
        data,
        (n_detectors, n_frames, pixel_size, pixel_size),
        src / "images.parquet",
    )
    _write_scalars_parquet(n_frames, src / "scalars.parquet")

    cache = tmp_path / "cache_scrambled_det"
    with pytest.raises(ValueError, match="not sequential"):
        rewrite_v3_directory(src, cache, frames_per_chunk=3)


def test_rewrite_rejects_detector_out_of_order(tmp_path: Path):
    """
    Across detectors, the (detector, frame) lex sequence must be
    non-decreasing -- i.e. all of det 0 before any of det 1.  A run
    that revisits an earlier detector must be rejected.
    """
    n_frames = 6
    n_detectors = 3
    pixel_size = 32
    # det 0 fully, then det 2, then back to det 1 -- the last segment
    # is the violation.
    detector = np.array([0, 0, 2, 2, 1, 1], dtype=np.uint8)
    frame = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint32)
    row = np.zeros(6, dtype=np.uint16)
    col = np.zeros(6, dtype=np.uint16)
    data = np.arange(1, 7, dtype=np.uint32)

    src = tmp_path / "run_scrambled_det_order"
    src.mkdir()
    _write_raw_images_parquet(
        detector,
        frame,
        row,
        col,
        data,
        (n_detectors, n_frames, pixel_size, pixel_size),
        src / "images.parquet",
    )
    _write_scalars_parquet(n_frames, src / "scalars.parquet")

    cache = tmp_path / "cache_scrambled_det_order"
    with pytest.raises(ValueError, match="not sequential"):
        rewrite_v3_directory(src, cache, frames_per_chunk=3)


def test_rewrite_idempotent_when_unchanged(
    tmp_path: Path, synth_v3_dir: tuple[Path, sparse.COO, dict]
):
    src, _, _ = synth_v3_dir
    cache = tmp_path / "cache"
    rewrite_v3_directory(src, cache, frames_per_chunk=3)
    # Collect mtimes of every cache file.
    before = {p: p.stat().st_mtime_ns for p in cache.iterdir()}
    # Pause to ensure mtime resolution would notice a rewrite.
    time.sleep(0.05)
    res = rewrite_v3_directory(src, cache, frames_per_chunk=3)
    assert res.skipped
    after = {p: p.stat().st_mtime_ns for p in cache.iterdir()}
    assert before == after


def test_rewrite_redo_when_chunk_size_changes(
    tmp_path: Path, synth_v3_dir: tuple[Path, sparse.COO, dict]
):
    src, _, _ = synth_v3_dir
    cache = tmp_path / "cache"
    rewrite_v3_directory(src, cache, frames_per_chunk=3)
    res = rewrite_v3_directory(src, cache, frames_per_chunk=4)
    assert not res.skipped
    assert res.chunks == (4, 3)


def test_rewrite_recovers_from_deleted_cache(
    tmp_path: Path, synth_v3_dir: tuple[Path, sparse.COO, dict]
):
    src, _, _ = synth_v3_dir
    cache = tmp_path / "cache"
    first = rewrite_v3_directory(src, cache, frames_per_chunk=3)
    # Nuke the cache.
    for p in cache.iterdir():
        p.unlink()
    second = rewrite_v3_directory(src, cache, frames_per_chunk=3)
    assert not second.skipped
    assert second.chunks == first.chunks
    assert (cache / LAYOUT.manifest_name).exists()


def test_rewrite_rejects_overlap(
    synth_v3_dir: tuple[Path, sparse.COO, dict],
):
    src, _, _ = synth_v3_dir
    with pytest.raises(ValueError, match="cache_dir must differ"):
        rewrite_v3_directory(src, src)


# ---------------------------------------------------------------------------
# Tiled registration (in-process)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiled_client(tmp_path: Path):
    """Yield a tiled client backed by an in-memory catalog.

    The whole *tmp_path* is registered as both writable and readable
    storage so external file assets created elsewhere in the test (the
    rewritten cache, the original v3 dir) can be served.
    """
    from tiled.catalog import in_memory
    from tiled.client import Context, from_context
    from tiled.server.app import build_app

    storage_dir = tmp_path / "tiled_storage"
    storage_dir.mkdir()
    catalog = in_memory(
        writable_storage=str(storage_dir),
        readable_storage=[str(tmp_path)],
    )
    app = build_app(catalog)
    with Context.from_app(app) as ctx:
        yield from_context(ctx)


def test_register_run_round_trip(
    tmp_path: Path,
    synth_v3_dir: tuple[Path, sparse.COO, dict],
    tiled_client,
):
    src, _, md = synth_v3_dir
    cache = tmp_path / "cache"
    run_node = register_run(
        tiled_client,
        run_key="run_017",
        source_dir=src,
        cache_dir=cache,
        frames_per_chunk=3,
    )

    # Run-level metadata.
    meta = run_node.metadata
    assert meta["format_version"] == LAYOUT.format_version
    assert meta["n_frames"] == 7
    assert meta["shape"] == [
        DEFAULT_N_DETECTORS,
        7,
        DEFAULT_PIXEL_SIZE,
        DEFAULT_PIXEL_SIZE,
    ]
    assert meta["stream_names"] == ["primary", "baseline"]
    assert meta["nominal_bin"] == pytest.approx(0.001)
    assert meta["scan_md"] == md["scan_md"]

    # Primary stream contents.
    primary = run_node["primary"]
    assert "scalars" in primary
    assert sorted(k for k in primary if k.startswith("det_")) == [
        f"det_{i:02d}" for i in range(1, DEFAULT_N_DETECTORS + 1)
    ]

    # Sparse round-trip on every detector.
    reader = HRDRawV3(src)
    for det in range(1, DEFAULT_N_DETECTORS + 1):
        expected = reader.get_detector(det).todense()
        served = primary[f"det_{det:02d}"].read()
        assert np.array_equal(served.todense(), expected)

    # Scalars round-trip.
    scalars_df = primary["scalars"].read()
    assert {"tth", "monitor"}.issubset(scalars_df.columns)
    assert len(scalars_df) == 7

    # Baseline.
    baseline_df = run_node["baseline"]["readings"].read()
    assert len(baseline_df) == 2
    assert "PV:A" in baseline_df.columns


def test_register_root_mirrors_directory_structure(
    tmp_path: Path,
    synth_v3_dir: tuple[Path, sparse.COO, dict],
    tiled_client,
):
    src, _, _ = synth_v3_dir
    # Move src into a nested layout so register_root has to descend.
    root = tmp_path / "root"
    nested = root / "2026-01" / "session_A"
    nested.mkdir(parents=True)
    target = nested / src.name
    src.rename(target)
    cache_root = tmp_path / "cache_root"

    registered = register_root(tiled_client, root, cache_root, frames_per_chunk=3)
    assert registered == [target]

    node = tiled_client["2026-01"]["session_A"]["run_017"]
    assert node.metadata["format_version"] == LAYOUT.format_version
    # Cache mirrored too.
    assert (
        cache_root / "2026-01" / "session_A" / "run_017" / LAYOUT.manifest_name
    ).exists()


def test_register_run_replace(
    tmp_path: Path,
    synth_v3_dir: tuple[Path, sparse.COO, dict],
    tiled_client,
):
    src, _, _ = synth_v3_dir
    cache = tmp_path / "cache"
    register_run(
        tiled_client,
        run_key="run_017",
        source_dir=src,
        cache_dir=cache,
        frames_per_chunk=3,
    )
    # Re-register with replace=True (without it, this would error).
    register_run(
        tiled_client,
        run_key="run_017",
        source_dir=src,
        cache_dir=cache,
        frames_per_chunk=3,
        replace=True,
    )
    assert "run_017" in tiled_client


# ---------------------------------------------------------------------------
# CLI: init-config
# ---------------------------------------------------------------------------


def test_init_config_writes_yaml(tmp_path: Path):
    source_root = tmp_path / "src_root"
    cache_root = tmp_path / "cache_root"
    source_root.mkdir()
    cache_root.mkdir()
    out = tmp_path / "config.yml"
    db = tmp_path / "catalog.db"

    rc = cli.main(
        [
            "init-config",
            "--source-root",
            str(source_root),
            "--cache-root",
            str(cache_root),
            "--out",
            str(out),
            "--catalog-db",
            str(db),
        ]
    )
    assert rc == 0
    assert out.exists()

    with out.open("r") as fin:
        parsed = yaml.safe_load(fin)
    assert parsed["authentication"]["allow_anonymous_access"] is True
    assert parsed["authentication"]["single_user_api_key"] == "${TILED_API_KEY}"

    tree = parsed["trees"][0]
    assert tree["tree"] == "catalog"
    assert tree["path"] == "/"
    args = tree["args"]
    # uri must be a SQLAlchemy sqlite URL (NOT a generic file:// URI;
    # SQLAlchemy would try to load a 'file' dialect plugin and fail).
    assert args["uri"].startswith("sqlite:///")
    assert args["uri"].endswith(str(db.resolve()))
    # readable_storage entries are absolute paths.
    rs = args["readable_storage"]
    assert str(source_root.resolve()) in rs
    assert str(cache_root.resolve()) in rs
    for entry in rs:
        assert Path(entry).is_absolute()
    assert args["init_if_not_exists"] is True


def test_init_config_creates_parent_dirs(tmp_path: Path):
    source_root = tmp_path / "src_root"
    cache_root = tmp_path / "cache_root"
    source_root.mkdir()
    cache_root.mkdir()
    out = tmp_path / "nested" / "deeper" / "config.yml"
    db = tmp_path / "db_nested" / "catalog.db"
    assert not out.parent.exists()
    assert not db.parent.exists()

    rc = cli.main(
        [
            "init-config",
            "--source-root",
            str(source_root),
            "--cache-root",
            str(cache_root),
            "--out",
            str(out),
            "--catalog-db",
            str(db),
        ]
    )
    assert rc == 0
    assert out.exists()
    assert out.parent.is_dir()
    assert db.parent.is_dir()
    # The catalog db itself is not created by init-config (tiled will
    # init it lazily).
    assert not db.exists()


def test_init_config_refuses_overwrite(tmp_path: Path):
    source_root = tmp_path / "src_root"
    cache_root = tmp_path / "cache_root"
    source_root.mkdir()
    cache_root.mkdir()
    out = tmp_path / "config.yml"
    db = tmp_path / "catalog.db"

    common = [
        "init-config",
        "--source-root",
        str(source_root),
        "--cache-root",
        str(cache_root),
        "--out",
        str(out),
        "--catalog-db",
        str(db),
    ]
    assert cli.main(common) == 0
    # Second invocation without --force fails.
    assert cli.main(common) == 2
    # With --force it succeeds.
    assert cli.main([*common, "--force"]) == 0


def test_init_config_rejects_nested_paths(tmp_path: Path):
    base = tmp_path / "base"
    inner = base / "inner"
    base.mkdir()
    inner.mkdir()
    out = tmp_path / "config.yml"
    db = tmp_path / "catalog.db"

    # cache inside source
    rc = cli.main(
        [
            "init-config",
            "--source-root",
            str(base),
            "--cache-root",
            str(inner),
            "--out",
            str(out),
            "--catalog-db",
            str(db),
        ]
    )
    assert rc == 2
    assert not out.exists()

    # source inside cache
    rc = cli.main(
        [
            "init-config",
            "--source-root",
            str(inner),
            "--cache-root",
            str(base),
            "--out",
            str(out),
            "--catalog-db",
            str(db),
        ]
    )
    assert rc == 2

    # Same path
    rc = cli.main(
        [
            "init-config",
            "--source-root",
            str(base),
            "--cache-root",
            str(base),
            "--out",
            str(out),
            "--catalog-db",
            str(db),
        ]
    )
    assert rc == 2
