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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass

from hrd_tools.detector_stats import Detector, detectors
from multihead.corrections import tth_from_z
from multihead.config import AnalyzerConfig

plt.switch_backend('qtagg')
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
    crystal_to_detector: float = 120.0,
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
        total_distance - crystal_to_detector,  # R: sample to analyzer
        crystal_to_detector,                   # Rd: analyzer to detector
        np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),  # theta_B (Bragg angle)
        2 * np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),  # tth_B (2*Bragg angle)
        detector_roll=0,
    )
    
    # Generate 2theta angles
    twotheta_vals = np.linspace(twotheta_range[0], twotheta_range[1], n_steps)
    
    # Convert pixel pitch from µm to mm
    pixel_size_mm = detector.pixel_pitch / 1000.0
    
    # Create array of pixel positions from center to edge
    n_pixels_half = detector.sensor_shape[0] // 2
    pixel_positions = np.arange(0, n_pixels_half + 1) * pixel_size_mm
    
    # Vectorized calculation: tth_from_z can handle both z and arm_tth arrays
    # Shape: (n_tth, n_positions)
    (tth_vals), (phi_vals) = tth_from_z(
        pixel_positions.reshape(1, -1),
        twotheta_vals.reshape(-1, 1),
        config,
    )
    
    # Maximum phi is at the detector edge (last pixel) for each 2theta
    max_phi = np.abs(phi_vals[:, -1])
    
    # Calculate uncertainty as the 2theta difference across each pixel
    # Shape: (n_tth, n_positions-1)
    pixel_uncertainties = np.abs(np.diff(tth_vals, axis=1))
    
    # Find where pixel uncertainty crosses threshold for each 2theta
    # Use argmax to find first True value (crossing threshold)
    # argmax returns 0 if all False, so check if any values exceed threshold
    crosses_threshold = pixel_uncertainties > error_threshold
    first_crossing = np.argmax(crosses_threshold, axis=1)
    
    # If no crossing (all False), argmax returns 0, so use max_phi
    # Otherwise use phi at first crossing position
    error_phi = np.where(
        np.any(crosses_threshold, axis=1),
        np.abs(phi_vals[np.arange(n_steps), first_crossing]),
        max_phi
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
# Select detector
detector = detectors["medipix4"]

# Distance sweep parameters
distances = np.linspace(1000, 1500, 9)  # mm
results_list = []

for dist in distances:
    result = analyze_detector_geometry(
        dist,
        detector,
        crystal_to_detector=120.0,
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

# Plot 1: Maximum Phi vs 2theta for different distances
ax1 = axes[0]
for result in results_list:
    ax1.plot(result.twotheta, result.max_phi, 
             label=f"{result.total_distance:.0f} mm")

ax1.set_xlabel('Arm 2θ (degrees)', fontsize=12)
ax1.set_ylabel(r'Maximum $\phi$ (degrees)', fontsize=12)
ax1.set_title(r'Maximum $\phi$ Coverage' + f'\n{detector.name}', fontsize=13)
ax1.legend(title='Total Distance', fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Error-limited Phi vs 2theta
ax2 = axes[1]
for result in results_list:
    ax2.plot(result.twotheta, result.error_threshold_phi, 
             label=f"{result.total_distance:.0f} mm")

ax2.set_xlabel('Arm 2θ (degrees)', fontsize=12)
ax2.set_ylabel(r'Error-Limited $\phi$ (degrees)', fontsize=12)
ax2.set_title(r'$\phi$ at Error Threshold (1e-4)' + f'\n{detector.name}', fontsize=13)
ax2.legend(title='Total Distance', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# ## Distance Effect Summary
# 
# Plot how the maximum phi varies with total distance at specific 2theta values.

# %%
fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

# Extract specific 2theta values
twotheta_targets = [30, 60, 90, 120]
colors = mpl.colormaps['viridis'](np.linspace(0, 1, len(twotheta_targets)))

for twotheta_target, color in zip(twotheta_targets, colors):
    max_phi_at_target = []
    error_phi_at_target = []
    
    for result in results_list:
        # Find closest 2theta value
        idx = np.argmin(np.abs(result.twotheta - twotheta_target))
        max_phi_at_target.append(result.max_phi[idx])
        error_phi_at_target.append(result.error_threshold_phi[idx])
    
    ax.plot(distances, max_phi_at_target, 'o-', color=color, 
            label=f'2θ = {twotheta_target}° (max)', linewidth=2)
    ax.plot(distances, error_phi_at_target, 's--', color=color, 
            label=f'2θ = {twotheta_target}° (error limit)', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Total Distance (mm)', fontsize=12)
ax.set_ylabel(r'$\phi$ Coverage (degrees)', fontsize=12)
ax.set_title(r'Effect of Total Distance on $\phi$ Coverage' + f'\n{detector.name}', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.show()

# %% [markdown]
# ## Compare Detectors
# 
# Compare different detector types at a fixed distance.

# %%
fixed_distance = 1000.0  # mm
detector_comparison = {}

for det_name, det in detectors.items():
    result = analyze_detector_geometry(
        fixed_distance,
        det,
        crystal_to_detector=120.0,
        twotheta_range=(1, 90),
        n_steps=512,
        error_threshold=1e-4,
    )
    detector_comparison[det_name] = result

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")

# Plot detector comparison
ax1 = axes[0]
for det_name, result in detector_comparison.items():
    ax1.plot(result.twotheta, result.max_phi, 
             label=result.detector_name, linewidth=2)

ax1.set_xlabel('Arm 2θ (degrees)', fontsize=12)
ax1.set_ylabel(r'Maximum $\phi$ (degrees)', fontsize=12)
ax1.set_title(r'Detector Comparison: Maximum $\phi$' + f'\nTotal Distance = {fixed_distance} mm', 
              fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot error-limited comparison
ax2 = axes[1]
for det_name, result in detector_comparison.items():
    ax2.plot(result.twotheta, result.error_threshold_phi, 
             label=result.detector_name, linewidth=2)

ax2.set_xlabel('Arm 2θ (degrees)', fontsize=12)
ax2.set_ylabel(r'Error-Limited $\phi$ (degrees)', fontsize=12)
ax2.set_title(r'Detector Comparison: Error-Limited $\phi$' + f'\nTotal Distance = {fixed_distance} mm', 
              fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.show()

# %%
