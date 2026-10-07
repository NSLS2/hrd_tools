"""Cross-correlation of AnalyzerConfig calibration errors on corrected Δ2θ.

``mac_sensitivity.py`` computes the maximum deviation of *one* parameter at a
time, assuming everything else is ideal.  Some parameters, however, only
matter when another parameter is already non-zero (e.g. crystal yaw has no
effect unless there is a non-zero crystal roll).

This script sweeps every *pair* of parameters simultaneously over the ranges
declared in ``beamline_params.yaml`` under ``parameter_bounds`` and measures
the resulting scatter ``Δ2θ`` (using the same ``tth_from_z`` correction model
as ``mac_sensitivity.py``).  For each pair it reports:

* the worst-case ``|Δ2θ|`` over the 2D grid, and
* the *coupling* term -- worst-case minus the linear sum of each parameter's
  individual worst-case -- which isolates genuine cross-correlation.

It also saves a triangular grid of pairwise ``Δ2θ`` heatmaps.  Cells where the
correction exceeds the tolerance are drawn in a saturated "over" color so the
unstable region is obvious at a glance.

Run with::

    pixi run -e xrt python design_scripts/FDR/mac_crosscorrelation.py
"""

from __future__ import annotations

import dataclasses
import itertools

import _fdr_params
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from multihead.corrections import tth_from_z

from hrd_tools.sensitivity import PARAM_METADATA, create_default_config

_args = _fdr_params.parse_args(__doc__)

MAX_DELTA_TTH_MDEG = 0.01  # mdeg -- tolerance threshold
Z_D = 15  # mm -- detector z position
ARM_ANGLES = [5, 88]  # deg
GRID_N = 41  # samples per axis on each pairwise grid

# Parameters excluded from the cross-correlation study: their tolerances are
# set far more strictly by other factors (theta_i by ~1/5 of the Darwin width,
# theta_d by keeping the scatter on a fixed detector pixel), so they only
# clutter the pairwise grid here.
EXCLUDE_PARAMS = {"theta_i", "theta_d"}

# Map the ``parameter_bounds`` YAML keys onto AnalyzerConfig field names.
# The YAML uses physics-y short names; AnalyzerConfig uses descriptive ones.
YAML_TO_FIELD = {
    "R": "R",
    "Rd": "Rd",
    "theta_i": "theta_i",
    "chi_c": "crystal_roll",
    "yaw_c": "crystal_yaw",
    "theta_d": "theta_d",
    "yaw_d": "detector_yaw",
    "roll_d": "detector_roll",
    "z": "center",
}


def _bounds() -> dict[str, tuple[float, float]]:
    """Read the ``parameter_bounds`` block from the blessed YAML.

    Parameters in :data:`EXCLUDE_PARAMS` are dropped.
    """
    raw = _fdr_params.load(_args.__dict__.get("params_file"))
    return {
        k: (float(lo), float(hi))
        for k, (lo, hi) in raw["parameter_bounds"].items()
        if k not in EXCLUDE_PARAMS
    }


# A couple of PARAM_METADATA unicode symbols use codepoints missing from the
# default Matplotlib font (e.g. the modifier-letter subscript in ψ꜀); override
# them with portable equivalents for clean labels.
_SYMBOL_OVERRIDE = {
    "crystal_yaw": "ψ_c",
}


def _symbol(yaml_key: str) -> str:
    """Unicode label for a YAML parameter key (falls back to the key)."""
    field = YAML_TO_FIELD[yaml_key]
    if field in _SYMBOL_OVERRIDE:
        return _SYMBOL_OVERRIDE[field]
    meta = PARAM_METADATA.get(field)
    return meta["unicode"] if meta is not None else yaml_key


def _units(yaml_key: str) -> str:
    field = YAML_TO_FIELD[yaml_key]
    meta = PARAM_METADATA.get(field)
    return meta["units"] if meta is not None else ""


def _delta_tth(
    base_config,
    base_tth,
    deltas: dict[str, float],
    arm_angle: float,
) -> float:
    """|Δ2θ| in mdeg for a set of simultaneous parameter deltas."""
    replacements = {
        YAML_TO_FIELD[k]: getattr(base_config, YAML_TO_FIELD[k]) + v
        for k, v in deltas.items()
    }
    perturbed = dataclasses.replace(base_config, **replacements)
    corrected, _ = tth_from_z(Z_D, arm_angle, perturbed)
    return float(abs(corrected - base_tth)) * 1000.0


def _single_worst(base_config, base_tth, yaml_key, lo, hi, arm_angle) -> float:
    """Worst-case |Δ2θ| (mdeg) varying a single parameter over its bound."""
    grid = np.linspace(lo, hi, GRID_N)
    return max(
        _delta_tth(base_config, base_tth, {yaml_key: d}, arm_angle) for d in grid
    )


def coupling_matrix(base_config, keys, bounds_map, arm_angles, grid_n=21):
    """Symmetric matrix of cross-coupling magnitude between every parameter pair.

    The coupling for a pair (A, B) is the worst-case (over a ``grid_n`` by
    ``grid_n`` grid of their bounds and over ``arm_angles``) of

        | dtth(dA, dB) - [ dtth(dA, 0) + dtth(0, dB) ] |

    i.e. how far the joint effect departs from the sum of the two parameters
    acting independently.  Zero means the parameters are separable
    (no cross-correlation); larger means they are coupled.
    """
    m = len(keys)
    mat = np.zeros((m, m))
    for arm in arm_angles:
        base_tth, _ = tth_from_z(Z_D, arm, base_config)
        grids = {k: np.linspace(*bounds_map[k], grid_n) for k in keys}
        # Cache the single-parameter response along each axis value.
        singles = {
            k: np.array(
                [_delta_tth(base_config, base_tth, {k: d}, arm) for d in grids[k]]
            )
            for k in keys
        }
        for i, j in itertools.combinations(range(m), 2):
            a, b = keys[i], keys[j]
            coup = 0.0
            for ia, da in enumerate(grids[a]):
                ea = singles[a][ia]
                for ib, db in enumerate(grids[b]):
                    comb = _delta_tth(base_config, base_tth, {a: da, b: db}, arm)
                    coup = max(coup, abs(comb - (ea + singles[b][ib])))
            mat[i, j] = mat[j, i] = max(mat[i, j], coup)
    return mat


def coupling_order(keys, mat):
    """Order ``keys`` so strongly-coupled parameters are adjacent.

    Uses average-linkage hierarchical clustering with SciPy optimal-leaf
    ordering on a distance derived from the coupling matrix (coupled -> near).
    """
    cmax = mat.max()
    if cmax <= 0:
        return list(keys)
    dist = 1.0 - (mat / cmax)
    np.fill_diagonal(dist, 0.0)

    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    cond = squareform(dist, checks=False)
    z = optimal_leaf_ordering(linkage(cond, method="average"), cond)
    return [keys[i] for i in leaves_list(z)]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_blessed = _fdr_params.complete_config()
_e_keV = (
    _args.energy_keV
    if _args.energy_keV is not None
    else _blessed.source.E_incident / 1000.0
)
config = create_default_config(_e_keV)

bounds = _bounds()
params = list(bounds)

# Reorder the parameters so that strongly cross-coupled parameters sit next to
# each other (and separable ones are pushed apart).  This makes the coupling
# blocks read as contiguous regions in the printed table and the heatmap grid.
_coupling = coupling_matrix(config, params, bounds, ARM_ANGLES)
params = coupling_order(params, _coupling)
print("Parameter order (coupled adjacent): " + ", ".join(_symbol(p) for p in params))
print()

# A colormap that makes threshold exceedance obvious: a perceptually-uniform
# ramp for in-tolerance values, with a saturated magenta "over" color for any
# cell above MAX_DELTA_TTH_MDEG.
cmap = mpl.colormaps["viridis"].with_extremes(over="magenta")
norm = mcolors.Normalize(vmin=0.0, vmax=MAX_DELTA_TTH_MDEG)


# ---------------------------------------------------------------------------
# Printed worst-case / coupling table
# ---------------------------------------------------------------------------

print(
    f"Cross-correlation of calibration errors (z_d={Z_D}mm, "
    f"threshold Δ2θ ≤ {MAX_DELTA_TTH_MDEG} mdeg)\n"
)
print("Worst-case |Δ2θ| over each parameter pair's bounds, and the coupling")
print("term (combined worst-case minus the sum of each acting alone).\n")

for arm_angle in ARM_ANGLES:
    base_tth, _ = tth_from_z(Z_D, arm_angle, config)

    # Per-parameter worst-case so we can subtract it out for the coupling term.
    single = {
        k: _single_worst(config, base_tth, k, *bounds[k], arm_angle) for k in params
    }

    print(f"2Θ = {arm_angle:>2}°")
    header = (
        f"  {'pair':<18} {'combined':>10} {'A-alone':>9} {'B-alone':>9} {'coupling':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for a, b in itertools.combinations(params, 2):
        ga = np.linspace(*bounds[a], GRID_N)
        gb = np.linspace(*bounds[b], GRID_N)
        combined = 0.0
        for da in ga:
            for db in gb:
                combined = max(
                    combined,
                    _delta_tth(config, base_tth, {a: da, b: db}, arm_angle),
                )
        coupling = combined - (single[a] + single[b])
        flag = "  <-- exceeds" if combined > MAX_DELTA_TTH_MDEG else ""
        pair = f"{_symbol(a)}×{_symbol(b)}"  # noqa: RUF001
        print(
            f"  {pair:<18} {combined:>10.3g} {single[a]:>9.3g} "
            f"{single[b]:>9.3g} {coupling:>9.3g}{flag}"
        )
    print()


# ---------------------------------------------------------------------------
# Pairwise heatmap grid (one figure per arm angle)
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt  # noqa: E402  (deferred so --no-show still prints)

save = _fdr_params.figure_saver(_args)
n = len(params)

_grids = {p: np.linspace(*bounds[p], GRID_N) for p in params}

for arm_angle in ARM_ANGLES:
    base_tth, _ = tth_from_z(Z_D, arm_angle, config)

    # Precompute the Δ2θ map for each unique pair once; the upper triangle
    # reuses the transpose of the lower-triangle data (technically redundant,
    # but it lets the eye read either parameter as the x axis).
    pair_grids: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n):
        for j in range(i + 1, n):
            px, py = params[j], params[i]  # x = params[j], y = params[i]
            gx, gy = _grids[px], _grids[py]
            grid = np.empty((GRID_N, GRID_N))
            for iy, dy in enumerate(gy):
                for ix, dx in enumerate(gx):
                    grid[iy, ix] = _delta_tth(
                        config, base_tth, {px: dx, py: dy}, arm_angle
                    )
            pair_grids[(i, j)] = grid

    fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n), layout="constrained")
    fig.suptitle(
        rf"Cross-correlation $\Delta 2\theta$ (mdeg) — $2\Theta$={arm_angle}°, "
        rf"$z_d$={Z_D}mm  (over→magenta @ {MAX_DELTA_TTH_MDEG} mdeg)"
    )

    mappable = None
    for r in range(n):  # row -> y-axis parameter params[r]
        py = params[r]
        for c in range(n):  # col -> x-axis parameter params[c]
            ax = axes[r][c]
            px = params[c]
            gx = _grids[px]
            gy = _grids[py]

            if r == c:
                # Diagonal: single-parameter Δ2θ vs deviation.
                vals = np.array(
                    [_delta_tth(config, base_tth, {px: d}, arm_angle) for d in gx]
                )
                ax.plot(gx, vals, color="C0")
                ax.axhline(MAX_DELTA_TTH_MDEG, color="magenta", ls="--", lw=0.8)
                ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=6)
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(r"$|\Delta 2\theta|$", fontsize=6)
                ax.set_facecolor("0.97")
            else:
                # Off-diagonal: pairwise heatmap. Grids are stored keyed by
                # (i, j) with i < j, where axis-0 (rows) is params[i] and
                # axis-1 (cols) is params[j]. For display in cell (r, c) we
                # need axis-0 -> y = params[r] and axis-1 -> x = params[c].
                #   * upper triangle (r < c): key=(r, c) already matches.
                #   * lower triangle (r > c): key=(c, r); transpose so the
                #     stored params[c]-rows/params[r]-cols become rows=params[r].
                key = (r, c) if r < c else (c, r)
                grid = pair_grids[key]
                if r > c:
                    grid = grid.T
                mappable = ax.imshow(
                    grid,
                    cmap=cmap,
                    norm=norm,
                    origin="lower",
                    aspect="auto",
                    extent=(gx[0], gx[-1], gy[0], gy[-1]),
                    interpolation="nearest",
                )
                ax.tick_params(labelsize=6)

            if c == 0:
                ax.set_ylabel(f"{_symbol(py)} ({_units(py)})", fontsize=8)
            if r == n - 1:
                ax.set_xlabel(f"{_symbol(px)} ({_units(px)})", fontsize=8)

    if mappable is not None:
        cb = fig.colorbar(mappable, ax=axes, extend="max", shrink=0.6)
        cb.set_label(r"$|\Delta 2\theta|$ (mdeg)")

    out = save(fig, f"mac_crosscorrelation_arm{arm_angle}.png")
    print(f"Saved {out}")

_fdr_params.maybe_show(_args)
