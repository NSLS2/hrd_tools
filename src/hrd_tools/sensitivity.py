"""
Sensitivity analysis for AnalyzerConfig parameters.

This module provides tools to analyze how errors in calibration parameters
affect the corrected 2θ values in a multi-analyzer spectrometer.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from multihead.config import AnalyzerConfig
from multihead.corrections import tth_from_z

from hrd_tools.xrt import CrystalProperties

# Parameter metadata: LaTeX label, Unicode symbol, units, default deltas, search bracket
PARAM_METADATA: dict[str, dict] = {
    "R": {
        "latex": r"$R$",
        "unicode": "R",
        "units": "mm",
        "default_deltas": [0.005, 0.05, 0.5],
        "bracket": [0, 100],
        "description": "sample to crystal distance",
    },
    "Rd": {
        "latex": r"$R_d$",
        "unicode": "R_d",
        "units": "mm",
        "default_deltas": [1, 5, 10],
        "bracket": [0, 100],
        "description": "crystal to detector distance",
    },
    "theta_i": {
        "latex": r"$\theta_i$",
        "unicode": "θᵢ",
        "units": "deg",
        "default_deltas": [0.001, 0.01, 0.1],
        "bracket": [0, 10],
        "description": "incident angle",
    },
    "theta_d": {
        "latex": r"$\theta_d$",
        "unicode": "θ_d",
        "units": "deg",
        "default_deltas": [0.1, 1, 10],
        "bracket": [0, 100],
        "description": "detector angle",
    },
    "crystal_roll": {
        "latex": r"$\chi$",
        "unicode": "χ",
        "units": "deg",
        "default_deltas": [0.0001, 0.005, 0.01],
        "bracket": [0, 0.1],
        "description": "crystal roll misalignment",
    },
    "crystal_yaw": {
        "latex": r"$\psi_c$",
        "unicode": "ψ꜀",
        "units": "deg",
        "default_deltas": [0.0001, 0.005, 0.01],
        "bracket": [0, 0.1],
        "description": "crystal yaw misalignment",
    },
    "detector_yaw": {
        "latex": r"$\psi_d$",
        "unicode": "ψᵈ",
        "units": "deg",
        "default_deltas": [0.0001, 0.005, 0.01],
        "bracket": [0, 0.1],
        "description": "detector yaw misalignment",
    },
    "detector_roll": {
        "latex": r"$\phi_d$",
        "unicode": "φᵈ",
        "units": "deg",
        "default_deltas": [0.0001, 0.005, 0.01],
        "bracket": [0, 0.1],
        "description": "detector roll misalignment",
    },
    "center": {
        "latex": r"$z_0$",
        "unicode": "z₀",
        "units": "mm",
        "default_deltas": [0.0055, 0.055, 2 * 0.055],
        "bracket": [0, 50],
        "description": "center offset",
    },
}


def _apply_delta(
    config: AnalyzerConfig, param_name: str, delta: float
) -> AnalyzerConfig:
    """Apply a delta to a single parameter in the config."""
    current_value = getattr(config, param_name)
    return dataclasses.replace(config, **{param_name: current_value + delta})


def _format_delta_label(param_name: str, delta: float) -> str:
    """Format a delta value for plot legends using LaTeX."""
    meta = PARAM_METADATA[param_name]
    units = meta["units"]
    latex = meta["latex"]

    # Convert to mdeg for angular units if small
    if units == "deg" and abs(delta) < 0.1:
        return rf"$\Delta${latex}={delta * 1000:.3g} mdeg"
    else:
        return rf"$\Delta${latex}={delta:.3g} {units}"


def _format_delta_unicode(param_name: str, delta: float) -> str:
    """Format a delta value for terminal output using Unicode."""
    meta = PARAM_METADATA[param_name]
    units = meta["units"]
    symbol = meta["unicode"]

    # Convert to mdeg for angular units if small
    if units == "deg" and abs(delta) < 0.1:
        return f"{symbol}={delta * 1000:.3g} mdeg"
    else:
        return f"{symbol}={delta:.3g} {units}"


def plot_sensitivity_vs_z(
    ax: Axes,
    base_config: AnalyzerConfig,
    param_name: str,
    deltas: Sequence[float],
    arm_angle: float = 45,
) -> None:
    """
    Plot sensitivity (Δ2θ) vs detector z position at a fixed arm angle.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    base_config : AnalyzerConfig
        Base analyzer configuration.
    param_name : str
        Name of the parameter to vary.
    deltas : Sequence[float]
        Delta values to apply to the parameter.
    arm_angle : float
        Fixed arm angle in degrees (default: 45).
    """
    z = np.linspace(-20, 20, 256)
    baseline, _ = tth_from_z(z, arm_angle, base_config)

    for delta in deltas:
        perturbed_config = _apply_delta(base_config, param_name, delta)
        corrected_tths, _ = tth_from_z(z, arm_angle, perturbed_config)
        label = _format_delta_label(param_name, delta)
        ax.plot(z, (baseline - corrected_tths) * 1000, label=label)

    ax.axhline(1e-1, color=".5", ls="--")
    ax.axhline(-1e-1, color=".5", ls="--")

    ax.legend()
    ax.set_title(rf"$2\Theta$={arm_angle}°")
    ax.set_xlabel(r"$z_d$ (mm)")


def plot_sensitivity_vs_arm_angle(
    ax: Axes,
    base_config: AnalyzerConfig,
    param_name: str,
    deltas: Sequence[float],
    z_fixed: float = 15,
) -> None:
    """
    Plot sensitivity (Δ2θ) vs arm angle at a fixed z position.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on.
    base_config : AnalyzerConfig
        Base analyzer configuration.
    param_name : str
        Name of the parameter to vary.
    deltas : Sequence[float]
        Delta values to apply to the parameter.
    z_fixed : float
        Fixed z position in mm (default: 15).
    """
    arm_angle = np.linspace(2, 88)
    baseline, _ = tth_from_z(z_fixed, arm_angle, base_config)

    for delta in deltas:
        perturbed_config = _apply_delta(base_config, param_name, delta)

        # Plot for +z
        corrected_tths, _ = tth_from_z(z_fixed, arm_angle, perturbed_config)
        (ln,) = ax.plot(
            arm_angle,
            (baseline - corrected_tths) * 1000,
            label=_format_delta_label(param_name, delta),
        )

        # Plot for -z (same color, no label)
        corrected_tths_neg, _ = tth_from_z(-z_fixed, arm_angle, perturbed_config)
        baseline_neg, _ = tth_from_z(-z_fixed, arm_angle, base_config)
        ax.plot(
            arm_angle, (baseline_neg - corrected_tths_neg) * 1000, color=ln.get_color()
        )

    ax.axhline(1e-1, color=".5", ls="--")
    ax.axhline(-1e-1, color=".5", ls="--")

    ax.set_title(rf"$z_d$={z_fixed}mm")
    ax.set_xlabel(r"arm $2\Theta$ (deg)")


def plot_sensitivity(
    fig: Figure,
    base_config: AnalyzerConfig,
    param_name: str,
    deltas: Sequence[float] | None = None,
    z_fixed: float = 15,
    arm_angle_fixed: float = 45,
) -> None:
    """
    Generate a two-panel sensitivity plot for a given parameter.

    Creates two subplots:
    - Left: Δ2θ vs z_d at a fixed arm angle
    - Right: Δ2θ vs arm angle at a fixed z_d

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to plot on.
    base_config : AnalyzerConfig
        Base analyzer configuration.
    param_name : str
        Name of the parameter to vary (must be in PARAM_METADATA).
    deltas : Sequence[float] | None
        Delta values to apply. If None, uses defaults from PARAM_METADATA.
    z_fixed : float
        Fixed z position for the right plot (default: 15 mm).
    arm_angle_fixed : float
        Fixed arm angle for the left plot (default: 45 deg).

    Raises
    ------
    ValueError
        If param_name is not a valid AnalyzerConfig parameter.
    """
    if param_name not in PARAM_METADATA:
        raise ValueError(
            f"Unknown parameter '{param_name}'. "
            f"Valid parameters: {list(PARAM_METADATA.keys())}"
        )

    if deltas is None:
        deltas = PARAM_METADATA[param_name]["default_deltas"]

    ax1, ax2 = fig.subplots(1, 2, sharey=True)

    plot_sensitivity_vs_z(ax1, base_config, param_name, deltas, arm_angle_fixed)
    plot_sensitivity_vs_arm_angle(ax2, base_config, param_name, deltas, z_fixed)

    ax1.set_ylabel(r"scatter $\Delta 2\theta$ (mdeg)")

    meta = PARAM_METADATA[param_name]
    fig.suptitle(f"Sensitivity to {meta['description']} ({meta['unicode']})")


def find_parameter_bound(
    base_config: AnalyzerConfig,
    param_name: str,
    max_delta_tth_mdeg: float = 0.1,
    z_d: float = 15,
    arm_angle: float = 88,
) -> float:
    """
    Find the maximum parameter deviation to keep Δ2θ below a threshold.

    Uses root-finding to determine the parameter value at which the
    change in corrected 2θ equals the specified absolute threshold.

    Parameters
    ----------
    base_config : AnalyzerConfig
        Base analyzer configuration.
    param_name : str
        Name of the parameter to vary.
    max_delta_tth_mdeg : float
        Maximum allowed Δ2θ in milli-degrees (default: 0.1).
    z_d : float
        Detector z position in mm (default: 15).
    arm_angle : float
        Arm angle in degrees (default: 88).

    Returns
    -------
    float
        Maximum parameter deviation that keeps Δ2θ below the threshold.

    Raises
    ------
    ValueError
        If param_name is not valid or root finding fails.
    """
    if param_name not in PARAM_METADATA:
        raise ValueError(
            f"Unknown parameter '{param_name}'. "
            f"Valid parameters: {list(PARAM_METADATA.keys())}"
        )

    max_delta_tth = max_delta_tth_mdeg / 1000
    baseline_tth, _ = tth_from_z(z_d, arm_angle, base_config)

    def objective(delta: float, target_delta_tth: float) -> float:
        perturbed_config = _apply_delta(base_config, param_name, delta)
        corrected_tth, _ = tth_from_z(z_d, arm_angle, perturbed_config)
        return target_delta_tth - abs(corrected_tth - baseline_tth)

    bracket = PARAM_METADATA[param_name]["bracket"]

    # Check if parameter has any sensitivity at all
    f_low = objective(bracket[0], max_delta_tth)
    f_high = objective(bracket[1], max_delta_tth)

    # If both have same sign, check if the parameter has negligible effect
    if f_low * f_high > 0:
        # Check sensitivity at bracket upper bound
        perturbed_config = _apply_delta(base_config, param_name, bracket[1])
        corrected_tth, _ = tth_from_z(z_d, arm_angle, perturbed_config)
        actual_delta = abs(corrected_tth - baseline_tth)

        if actual_delta < max_delta_tth:
            # Parameter has negligible effect even at large values
            raise ValueError(
                f"Parameter '{param_name}' has negligible sensitivity: "
                f"even at {bracket[1]} {PARAM_METADATA[param_name]['units']}, "
                f"Δ2θ = {actual_delta * 1000:.2g} mdeg < threshold {max_delta_tth_mdeg:.2g} mdeg"
            )

    try:
        result = optimize.root_scalar(objective, args=(max_delta_tth,), bracket=bracket)
        return result.root
    except ValueError as e:
        raise ValueError(
            f"Root finding failed for {param_name}: {e}. "
            f"Try adjusting the bracket {bracket}."
        ) from e


def print_bound_result(
    param_name: str,
    bound: float,
    z_d: float,
    arm_angle: float,
    max_delta_tth_mdeg: float,
) -> None:
    """Print the bound result using Unicode symbols."""
    meta = PARAM_METADATA[param_name]
    symbol = meta["unicode"]
    units = meta["units"]

    # Format bound value
    if units == "deg" and abs(bound) < 0.1:
        bound_str = f"{bound * 1000:.3g} mdeg"
    else:
        bound_str = f"{bound:.3g} {units}"

    print(
        f"At z_d={z_d}mm and 2Θ={arm_angle}° the maximum {symbol} "
        f"to stay under Δ2θ ≤ {max_delta_tth_mdeg:.2g} mdeg "
        f"is {bound_str}"
    )


def create_default_config(energy_kev: float = 40) -> AnalyzerConfig:
    """Create a default AnalyzerConfig for the given energy."""
    props = CrystalProperties.create(E=energy_kev)
    return AnalyzerConfig(
        R=950,  # sample to crystal distance (mm)
        Rd=120,  # crystal to detector distance (mm)
        theta_i=props.bragg_angle,
        theta_d=2 * props.bragg_angle,
        detector_roll=0,
    )


def main(args: list[str] | None = None) -> None:
    """CLI entry point for sensitivity analysis."""
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Analyze sensitivity of corrected 2θ to AnalyzerConfig parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -p crystal_roll --show
  %(prog)s -p theta_d -d 0.001,0.005,0.01 -o sensitivity.png
  %(prog)s -p R --tolerance 0.05 --show

Available parameters:
"""
        + "\n".join(f"  {k}: {v['description']}" for k, v in PARAM_METADATA.items()),
    )

    parser.add_argument(
        "-p",
        "--param",
        required=True,
        choices=list(PARAM_METADATA.keys()),
        help="Parameter to analyze",
    )
    parser.add_argument(
        "-d",
        "--deltas",
        type=str,
        default=None,
        help="Comma-separated delta values (default: use parameter defaults)",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.1,
        help="Max Δ2θ in mdeg (default: 0.1)",
    )
    parser.add_argument(
        "-E",
        "--energy",
        type=float,
        default=40,
        help="Energy in keV (default: 40)",
    )
    parser.add_argument(
        "-z",
        "--z-position",
        type=float,
        default=15,
        help="z_d position for bound calculation (default: 15 mm)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file for figure (optional)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting, only compute bound",
    )

    parsed = parser.parse_args(args)

    # Parse deltas
    deltas: list[float] | None = None
    if parsed.deltas:
        deltas = [float(x.strip()) for x in parsed.deltas.split(",")]

    # Create config
    config = create_default_config(parsed.energy)

    # Compute bounds at both low and high arm angles
    arm_angles = [5, 88]
    meta = PARAM_METADATA[parsed.param]
    symbol = meta["unicode"]
    units = meta["units"]

    print(
        f"Parameter bounds for {symbol} ({meta['description']}) "
        f"at z_d={parsed.z_position}mm, Δ2θ ≤ {parsed.tolerance:.2g} mdeg:"
    )

    for arm_angle in arm_angles:
        try:
            bound = find_parameter_bound(
                config,
                parsed.param,
                max_delta_tth_mdeg=parsed.tolerance,
                z_d=parsed.z_position,
                arm_angle=arm_angle,
            )
            # Format bound value
            if units == "deg" and abs(bound) < 0.1:
                bound_str = f"{bound * 1000:.3g} mdeg"
            else:
                bound_str = f"{bound:.3g} {units}"
            print(f"  2Θ={arm_angle:2}°: {symbol} ≤ {bound_str}")
        except ValueError as e:
            print(f"  2Θ={arm_angle:2}°: {e}")

    # Generate plot
    if not parsed.no_plot:
        fig = plt.figure(layout="constrained", figsize=(10, 5))
        plot_sensitivity(
            fig,
            config,
            parsed.param,
            deltas=deltas,
            z_fixed=parsed.z_position,
            arm_angle_fixed=45,
        )

        if parsed.output:
            fig.savefig(parsed.output, dpi=400)
            print(f"Figure saved to {parsed.output}")

        if parsed.show:
            plt.show()


if __name__ == "__main__":
    main()
