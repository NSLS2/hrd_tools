# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Detector Distance Study
# 
# This notebook studies the effect of varying the total distance from sample to detector
# on the maximum phi angle and the inherent error in diffraction measurements.

# %% [markdown]
# ## Import Libraries

# %%
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from multihead.config import AnalyzerConfig
from multihead.corrections import tth_from_z

import _fdr_params
from hrd_tools.detector_stats import Detector, detectors
from hrd_tools.xrt import CrystalProperties

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)
_blessed = _fdr_params.complete_config()

_e_keV = _args.energy_keV if _args.energy_keV is not None else _blessed.source.E_incident / 1000.0
_props = CrystalProperties.create(E=_e_keV)

# %% [markdown]
# ## Analysis Function
# 
# The main function computes for each arm 2theta angle:
# - Maximum phi visible on the detector
# - Phi where error crosses the threshold (or max phi if always below threshold)

# %%
@dataclass
class DetectorGeometryResult:
    """Results from detector geometry analysis."""

    twotheta: np.ndarray
    max_phi: np.ndarray
    error_threshold_phi: np.ndarray
    total_distance: float
    detector_name: str


# %%
def analyze_detector_geometry(
    total_distance: float,
    detector: Detector,
    *,
    crystal_to_detector: float = _blessed.analyzer.Rd,
    twotheta_range: tuple[float, float] = (0, 90),
    n_steps: int = 512,
    error_threshold: float = 1e-4,
) -> DetectorGeometryResult:
    """
    Analyze detector geometry for varying 2theta angles.
    
    Parameters
    ----------
    total_distance : float
        Total distance from sample to detector (mm)
    detector : Detector
        Detector object with specifications
    crystal_to_detector : float
        Distance from crystal to detector (mm), default 120mm
    twotheta_range : tuple[float, float]
        Range of arm 2theta angles (degrees), default (0, 120)
    n_steps : int
        Number of steps for analysis, default 512
    error_threshold : float
        Error threshold for phi cutoff, default 1e-4
        
    Returns
    -------
    DetectorGeometryResult
        Dataclass containing arrays of 2theta, max_phi, and error_threshold_phi
    """
    # Create configuration object for tth_from_z
    # R is sample to analyzer (crystal), Rd is analyzer to detector
    config = AnalyzerConfig(
        total_distance - crystal_to_detector,    # R: sample to analyzer
        crystal_to_detector,                     # Rd: analyzer to detector
        _props.bragg_angle,                      # theta_i (deg)
        2 * _props.bragg_angle,                  # theta_d (deg)
        detector_roll=_blessed.analyzer.roll,
    )

    twotheta_vals = np.linspace(twotheta_range[0], twotheta_range[1], n_steps)

    # Convert pixel pitch from µm to mm
    pixel_size_mm = detector.pixel_pitch / 1000.0

    n_pixels_half = detector.sensor_shape[0] // 2
    pixel_positions = np.arange(0, n_pixels_half + 1) * pixel_size_mm

    (tth_vals), (phi_vals) = tth_from_z(
        pixel_positions.reshape(1, -1),
        twotheta_vals.reshape(-1, 1),
        config,
    )

    max_phi = np.abs(phi_vals[:, -1])

    pixel_uncertainties = np.abs(np.diff(tth_vals, axis=1))

    crosses_threshold = pixel_uncertainties > error_threshold
    first_crossing = np.argmax(crosses_threshold, axis=1)

    error_phi = np.where(
        np.any(crosses_threshold, axis=1),
        np.abs(phi_vals[np.arange(n_steps), first_crossing]),
        max_phi,
    )

    return DetectorGeometryResult(
        twotheta=twotheta_vals,
        max_phi=max_phi,
        error_threshold_phi=error_phi,
        total_distance=total_distance,
        detector_name=detector.name,
    )


# %% [markdown]
# ## Parameter Sweep
# 
# Sweep through different total distances and analyze the effect on phi coverage.

# %%
# Use the blessed baseline detector
detector = _fdr_params.detector()

# Distance sweep parameters (mm)
distances = np.linspace(1000, 1500, 9)
results_list = []

for dist in distances:
    result = analyze_detector_geometry(
        dist,
        detector,
        twotheta_range=(1, 90),
        n_steps=512,
        error_threshold=1e-4,
    )
    results_list.append(result)

# %% [markdown]
# ## Visualization
# 
# Plot the maximum phi and error-limited phi as a function of 2theta for different distances.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

ax1 = axes[0]
for result in results_list:
    ax1.plot(result.twotheta, result.max_phi, label=f"{result.total_distance:.0f} mm")

ax1.set_xlabel("Arm 2θ (degrees)", fontsize=12)
ax1.set_ylabel(r"Maximum $\phi$ (degrees)", fontsize=12)
ax1.set_title(r"Maximum $\phi$ Coverage" + f"\n{detector.name}", fontsize=13)
ax1.legend(title="Total Distance", fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for result in results_list:
    ax2.plot(result.twotheta, result.error_threshold_phi, label=f"{result.total_distance:.0f} mm")

ax2.set_xlabel("Arm 2θ (degrees)", fontsize=12)
ax2.set_ylabel(r"Error-Limited $\phi$ (degrees)", fontsize=12)
ax2.set_title(r"$\phi$ at Error Threshold (1e-4)" + f"\n{detector.name}", fontsize=13)
ax2.legend(title="Total Distance", fontsize=9)
ax2.grid(True, alpha=0.3)

_save(fig, "detector_distance_phi_vs_tth.png")

# %% [markdown]
# ## Distance Effect Summary

# %%
fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

twotheta_targets = [30, 60, 90, 120]           # deg
colors = mpl.colormaps["viridis"](np.linspace(0, 1, len(twotheta_targets)))

for twotheta_target, color in zip(twotheta_targets, colors):
    max_phi_at_target = []
    error_phi_at_target = []

    for result in results_list:
        idx = np.argmin(np.abs(result.twotheta - twotheta_target))
        max_phi_at_target.append(result.max_phi[idx])
        error_phi_at_target.append(result.error_threshold_phi[idx])

    ax.plot(distances, max_phi_at_target, "o-", color=color,
            label=f"2θ = {twotheta_target}° (max)", linewidth=2)
    ax.plot(distances, error_phi_at_target, "s--", color=color,
            label=f"2θ = {twotheta_target}° (error limit)", linewidth=1.5, alpha=0.7)

ax.set_xlabel("Total Distance (mm)", fontsize=12)
ax.set_ylabel(r"$\phi$ Coverage (degrees)", fontsize=12)
ax.set_title(r"Effect of Total Distance on $\phi$ Coverage" + f"\n{detector.name}",
             fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

_save(fig, "detector_distance_effect.png")

# %% [markdown]
# ## Compare Detectors

# %%
fixed_distance = 1000.0                        # mm
detector_comparison = {}

for det_name, det in detectors.items():
    result = analyze_detector_geometry(
        fixed_distance,
        det,
        twotheta_range=(1, 90),
        n_steps=512,
        error_threshold=1e-4,
    )
    detector_comparison[det_name] = result

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

ax1 = axes[0]
for det_name, result in detector_comparison.items():
    ax1.plot(result.twotheta, result.max_phi, label=result.detector_name, linewidth=2)

ax1.set_xlabel("Arm 2θ (degrees)", fontsize=12)
ax1.set_ylabel(r"Maximum $\phi$ (degrees)", fontsize=12)
ax1.set_title(r"Detector Comparison: Maximum $\phi$" + f"\nTotal Distance = {fixed_distance} mm",
              fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for det_name, result in detector_comparison.items():
    ax2.plot(result.twotheta, result.error_threshold_phi, label=result.detector_name, linewidth=2)

ax2.set_xlabel("Arm 2θ (degrees)", fontsize=12)
ax2.set_ylabel(r"Error-Limited $\phi$ (degrees)", fontsize=12)
ax2.set_title(r"Detector Comparison: Error-Limited $\phi$" + f"\nTotal Distance = {fixed_distance} mm",
              fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

_save(fig, "detector_distance_comparison.png")
_fdr_params.maybe_show(_args)
