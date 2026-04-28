"""Tests for the FDR design-scripts shared parameter loader.

These tests exercise the YAML round-trip through ``hrd_tools.file_io`` and
the ``_fdr_params`` helper module that all of ``design_scripts/FDR/`` use to
read blessed beamline parameters.
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hrd_tools.config import (
    AnalyzerConfig,
    CompleteConfig,
    DetectorConfig,
    SimConfig,
    SimScanConfig,
    SourceConfig,
)
from hrd_tools.detector_stats import detectors
from hrd_tools.file_io import (
    complete_config_from_yaml,
    complete_config_to_yaml,
)

FDR_DIR = Path(__file__).resolve().parents[1] / "design_scripts" / "FDR"


@pytest.fixture(scope="module")
def fdr_params():
    """Import the (non-installed) ``_fdr_params`` helper module."""
    sys.path.insert(0, str(FDR_DIR))
    try:
        import _fdr_params  # type: ignore

        return _fdr_params
    finally:
        sys.path.remove(str(FDR_DIR))


# ---------------------------------------------------------------------------
# YAML round-trip on the file_io API
# ---------------------------------------------------------------------------


def _make_config(incident_angle: float = 1.234) -> CompleteConfig:
    return CompleteConfig(
        source=SourceConfig(
            E_incident=40000.0,
            pattern_path=None,
            dx=0.5,
            dz=0.5,
            dy=0.0,
            delta_phi=1.0,
            E_hwhm=0.0,
            v_div=0.0,
            h_div=0.0,
            min_tth=0.0,
            max_tth=180.0,
            source_offset_x=0.0,
            source_offset_y=0.0,
            source_offset_z=0.0,
        ),
        sim=SimConfig(nrays=1000),
        detector=DetectorConfig(pitch=0.055, transverse_size=256, height=14.08),
        analyzer=AnalyzerConfig(
            R=910.0,
            Rd=120.0,
            cry_offset=2.0,
            cry_width=30.0,
            cry_depth=70.0,
            N=12,
            incident_angle=incident_angle,
            thickness=10.0,
            roll=0.0,
        ),
        scan=SimScanConfig(start=0.0, stop=8.0, delta=1e-4, short_description="t"),
    )


def test_complete_config_yaml_round_trip_plain():
    cfg = _make_config()
    buf = io.StringIO()
    complete_config_to_yaml(cfg, buf)
    buf.seek(0)
    out = complete_config_from_yaml(buf)
    assert out == cfg


def test_complete_config_yaml_handles_numpy_scalars():
    """The derived Bragg angle is an ``np.float64``; it must serialize."""
    np = pytest.importorskip("numpy")
    cfg = _make_config(incident_angle=np.float64(2.833217163138173))
    buf = io.StringIO()
    complete_config_to_yaml(cfg, buf)
    buf.seek(0)
    out = complete_config_from_yaml(buf)
    assert out.analyzer.incident_angle == pytest.approx(2.833217163138173)


def test_complete_config_yaml_null_incident_angle_becomes_nan(tmp_path: Path):
    """``incident_angle: null`` round-trips as NaN so the loader can derive it."""
    yaml_text = """\
source:
  E_incident: 40000.0
  pattern_path: null
  dx: 0.5
  dz: 0.5
  dy: 0.0
  delta_phi: 1.0
  E_hwhm: 0.0
  v_div: 0.0
  h_div: 0.0
  min_tth: 0.0
  max_tth: 180.0
  source_offset_x: 0.0
  source_offset_y: 0.0
  source_offset_z: 0.0
sim:
  nrays: 1000
detector:
  pitch: 0.055
  transverse_size: 256
  height: 14.08
analyzer:
  R: 910.0
  Rd: 120.0
  cry_offset: 2.0
  cry_width: 30.0
  cry_depth: 70.0
  N: 12
  incident_angle: null
  thickness: 10.0
  roll: 0.0
scan:
  start: 0.0
  stop: 8.0
  delta: 1.0e-4
  short_description: "t"
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = complete_config_from_yaml(p)
    assert math.isnan(cfg.analyzer.incident_angle)


# ---------------------------------------------------------------------------
# _fdr_params loader
# ---------------------------------------------------------------------------


def test_loader_builds_complete_config(fdr_params):
    cfg = fdr_params.complete_config()
    assert isinstance(cfg, CompleteConfig)
    # Blessed values from beamline_params.yaml.
    assert cfg.source.E_incident == 40000.0
    assert cfg.analyzer.thickness == 10.0
    # incident_angle was null in the YAML; loader must derive a positive float.
    assert not math.isnan(cfg.analyzer.incident_angle)
    assert 0.0 < cfg.analyzer.incident_angle < 90.0


def test_loader_detector_block_from_named_asic(fdr_params):
    """The YAML carries only ``detector.name``; geometry comes from
    :mod:`hrd_tools.detector_stats`.
    """
    cfg = fdr_params.complete_config()
    rec = fdr_params.detector()
    assert rec is detectors[fdr_params.load()["detector"]["name"]]
    # mm == um/1000 and transverse_size == sensor_shape[1]
    assert cfg.detector.pitch == pytest.approx(rec.pixel_pitch / 1000.0)
    assert cfg.detector.transverse_size == rec.sensor_shape[1]
    assert cfg.detector.height == pytest.approx(
        rec.sensor_shape[1] * rec.pixel_pitch / 1000.0
    )


def test_loader_multihead_analyzer(fdr_params):
    pytest.importorskip("multihead")
    from multihead.config import AnalyzerConfig as MHAnalyzerConfig

    ac = fdr_params.analyzer_multihead()
    assert isinstance(ac, MHAnalyzerConfig)
    raw = fdr_params.load()["analyzer"]
    assert ac.R == raw["R"]
    assert ac.Rd == raw["Rd"]
    # theta_d should be 2 * theta_i (Bragg geometry).
    assert ac.theta_d == pytest.approx(2 * ac.theta_i)


def test_loader_multihead_analyzer_overrides(fdr_params):
    pytest.importorskip("multihead")
    ac = fdr_params.analyzer_multihead(
        energy_keV=25.0, R=500.0, Rd=530.0, crystal_roll=0.001
    )
    assert ac.R == 500.0
    assert ac.Rd == 530.0
    assert ac.crystal_roll == 0.001
    # 25 keV gives a bigger Bragg angle than 40 keV.
    ac40 = fdr_params.analyzer_multihead(energy_keV=40.0)
    assert ac.theta_i > ac40.theta_i


def test_loader_layout_block(fdr_params):
    layout = fdr_params.layout()
    for key in (
        "banks",
        "crystals_per_bank",
        "pitch_deg",
        "primary_offset_deg",
        "scan_range_deg",
        "frame_dt_s",
        "default_target_time_s",
    ):
        assert key in layout, f"missing layout key {key!r}"


def test_loader_rois_path_resolves(fdr_params):
    p = fdr_params.rois_path()
    assert p.exists(), f"ROI YAML not found at {p}"
    assert p.suffix in {".yaml", ".yml"}


def test_loader_rois_yaml_loads_with_multihead(fdr_params):
    pytest.importorskip("multihead")
    from multihead.config import DetectorROIs

    rois = DetectorROIs.from_yaml(fdr_params.rois_path())
    # 11-BM has 12 detectors; sanity-check the block is wired up.
    assert len(rois.rois) == 12


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def test_cli_argparser_defaults(fdr_params, tmp_path):
    parser = fdr_params.make_argparser(default_outdir=tmp_path)
    ns = parser.parse_args([])
    assert ns.outdir == tmp_path
    assert ns.dpi == 300
    assert ns.show is True
    assert ns.energy_kev is None


def test_cli_no_show_and_overrides(fdr_params, tmp_path):
    parser = fdr_params.make_argparser()
    ns = parser.parse_args(
        ["--no-show", "--outdir", str(tmp_path), "--dpi", "150", "--energy-kev", "25"]
    )
    assert ns.show is False
    assert ns.outdir == tmp_path
    assert ns.dpi == 150
    assert ns.energy_kev == 25.0


def test_figure_saver_writes_to_outdir(fdr_params, tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    args = fdr_params.CLIArgs(outdir=tmp_path / "figs", dpi=72, show=False, energy_keV=None)
    save = fdr_params.figure_saver(args)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save(fig, "test.png")
    plt.close(fig)
    assert out == tmp_path / "figs" / "test.png"
    assert out.exists()
    assert out.stat().st_size > 0
