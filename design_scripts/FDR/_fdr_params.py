"""Shared parameter loader and CLI helpers for design_scripts/FDR/.

Every FDR script imports from this module to:

* Load the blessed beamline parameters from ``beamline_params.yaml``
  (round-trippable through :func:`hrd_tools.file_io.complete_config_from_yaml`).
* Build the ``hrd_tools.config`` and ``multihead.config`` flavours of
  ``AnalyzerConfig`` from those parameters.
* Pick the baseline detector record from
  :data:`hrd_tools.detector_stats.detectors`.
* Get a uniform argparse CLI (``--outdir``, ``--dpi``, ``--show/--no-show``,
  ``--energy-kev``) so that scripts can be safely run pointing at a scratch
  output directory.

This module is intentionally **not** installed with ``hrd_tools``; it lives
next to the FDR scripts and is imported by relative path by them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from hrd_tools.config import CompleteConfig
from hrd_tools.detector_stats import Detector, detectors
from hrd_tools.file_io import complete_config_from_yaml
from hrd_tools.xrt import CrystalProperties

# Resolved path to the blessed YAML.
_HERE = Path(__file__).resolve().parent
DEFAULT_YAML = _HERE / "beamline_params.yaml"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _load_raw(path: Path | str | None = None) -> dict[str, Any]:
    """Return the full YAML document as a dict (incl. auxiliary blocks)."""
    p = Path(path) if path is not None else DEFAULT_YAML
    with open(p) as fh:
        return yaml.safe_load(fh)


def load(path: Path | str | None = None) -> dict[str, Any]:
    """Public alias for the raw YAML mapping."""
    return _load_raw(path)


def _expand_detector_block(raw: dict[str, Any]) -> dict[str, Any]:
    """Expand ``detector: {name: <key>}`` into the geometric fields that
    :class:`hrd_tools.config.DetectorConfig` expects, sourced from
    :data:`hrd_tools.detector_stats.detectors`.

    Returns a *new* mapping; ``raw`` is left untouched.
    """
    out = dict(raw)
    det_block = dict(out.get("detector", {}))
    name = det_block.get("name")
    if name is None:
        raise ValueError("YAML 'detector' block must specify 'name'.")
    rec = detectors[name]
    pitch_mm = rec.pixel_pitch / 1000.0  # µm -> mm
    transverse = rec.sensor_shape[1]
    det_block.setdefault("pitch", pitch_mm)
    det_block.setdefault("transverse_size", transverse)
    det_block.setdefault("height", transverse * pitch_mm)
    out["detector"] = det_block
    return out


def complete_config(path: Path | str | None = None) -> CompleteConfig:
    """Build a :class:`hrd_tools.config.CompleteConfig` from the blessed YAML.

    The ``detector`` block in the YAML carries only an ASIC ``name`` (a key
    into :data:`hrd_tools.detector_stats.detectors`); pitch/size/height are
    derived here so they stay single-sourced.

    The ``analyzer.incident_angle`` field is auto-derived from
    ``source.E_incident`` and the Si(111) Bragg angle when the YAML leaves
    it as ``null``.
    """
    import io
    import math

    import yaml

    raw = _expand_detector_block(_load_raw(path))
    cfg = complete_config_from_yaml(io.StringIO(yaml.safe_dump(raw)))
    if math.isnan(cfg.analyzer.incident_angle):
        e_keV = cfg.source.E_incident / 1000.0
        props = CrystalProperties.create(E=e_keV)
        cfg = replace(
            cfg, analyzer=replace(cfg.analyzer, incident_angle=props.bragg_angle)
        )
    return cfg


# ---------------------------------------------------------------------------
# Convenience accessors for the multihead-flavoured AnalyzerConfig used by
# the FDR scripts (and by hrd_tools.sensitivity).
# ---------------------------------------------------------------------------


def analyzer_multihead(
    energy_keV: float | None = None,
    *,
    path: Path | str | None = None,
    crystal_roll: float = 0.0,
    R: float | None = None,
    Rd: float | None = None,
):
    """Build a ``multihead.config.AnalyzerConfig`` from the blessed values.

    Parameters
    ----------
    energy_keV : float, optional
        Override the YAML's ``source.E_incident`` (in keV) used for the
        Bragg-angle derivation.
    path : str or Path, optional
        Override the YAML location.
    crystal_roll, R, Rd : float, optional
        Per-call overrides.
    """
    from multihead.config import AnalyzerConfig as MHAnalyzerConfig

    raw = _load_raw(path)
    a = raw["analyzer"]
    e_keV = energy_keV if energy_keV is not None else raw["source"]["E_incident"] / 1000.0
    props = CrystalProperties.create(E=e_keV)
    return MHAnalyzerConfig(
        R if R is not None else a["R"],
        Rd if Rd is not None else a["Rd"],
        props.bragg_angle,
        2 * props.bragg_angle,
        crystal_roll=crystal_roll,
        detector_roll=a.get("roll", 0.0),
    )


def detector(name: str | None = None, *, path: Path | str | None = None) -> Detector:
    """Return a baseline :class:`Detector` record.

    With ``name=None``, returns the YAML default (``detector.name``). Pass
    an explicit detector key to override.
    """
    raw = _load_raw(path)
    key = name or raw["detector"]["name"]
    return detectors[key]


# ---------------------------------------------------------------------------
# Auxiliary blocks
# ---------------------------------------------------------------------------


def layout(path: Path | str | None = None) -> dict[str, Any]:
    return _load_raw(path)["layout"]


def beam(path: Path | str | None = None) -> dict[str, Any]:
    return _load_raw(path)["beam"]


def crystal_reference(path: Path | str | None = None) -> dict[str, Any]:
    return _load_raw(path)["crystal_reference"]


def real_data(path: Path | str | None = None) -> dict[str, Any]:
    return _load_raw(path)["real_data"]


def rois_path(path: Path | str | None = None) -> Path:
    """Return the absolute path to the multihead ROIs YAML."""
    p = Path(path) if path is not None else DEFAULT_YAML
    rd = _load_raw(p)["real_data"]
    return (p.parent / rd["rois_file"]).resolve()


# ---------------------------------------------------------------------------
# Uniform CLI: --outdir, --dpi, --show/--no-show, --energy-kev
# ---------------------------------------------------------------------------


@dataclass
class CLIArgs:
    """Resolved CLI state."""

    outdir: Path
    dpi: int
    show: bool
    energy_keV: float | None


def make_argparser(
    description: str | None = None,
    *,
    default_outdir: str | Path = "./figures",
    default_dpi: int = 300,
    default_show: bool = True,
) -> argparse.ArgumentParser:
    """Return an ``ArgumentParser`` with the FDR-standard flags installed."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(default_outdir),
        help="Directory to save figures into (created if missing).",
    )
    parser.add_argument(
        "--dpi", type=int, default=default_dpi, help="DPI for saved figures."
    )
    show = parser.add_mutually_exclusive_group()
    show.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=default_show,
        help="Display figures interactively (default).",
    )
    show.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Skip plt.show; useful for automated runs.",
    )
    parser.add_argument(
        "--energy-kev",
        type=float,
        default=None,
        help="Override the canonical source energy (keV).",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="Override path to beamline_params.yaml.",
    )
    return parser


def parse_args(
    description: str | None = None, argv: list[str] | None = None
) -> CLIArgs:
    """Convenience: parse :data:`sys.argv` and return a :class:`CLIArgs`."""
    parser = make_argparser(description)
    ns = parser.parse_args(argv)
    return CLIArgs(
        outdir=Path(ns.outdir),
        dpi=int(ns.dpi),
        show=bool(ns.show),
        energy_keV=ns.energy_kev,
    )


def figure_saver(args: CLIArgs) -> Callable[[Any, str], Path]:
    """Return ``save(fig, name)`` -> Path that writes ``name`` under outdir."""

    args.outdir.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str) -> Path:
        out = args.outdir / name
        fig.savefig(out, dpi=args.dpi)
        return out

    return save


def maybe_show(args: CLIArgs, *, block: bool = True) -> None:
    """Call ``plt.show`` only when ``--show`` was set."""
    if not args.show:
        return
    import matplotlib.pyplot as plt

    plt.show(block=block)


__all__ = [
    "CLIArgs",
    "DEFAULT_YAML",
    "analyzer_multihead",
    "beam",
    "complete_config",
    "crystal_reference",
    "detector",
    "figure_saver",
    "layout",
    "load",
    "make_argparser",
    "maybe_show",
    "parse_args",
    "real_data",
    "rois_path",
]
