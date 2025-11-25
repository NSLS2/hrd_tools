# %% [markdown]
# # Ineherent correction error
#
# There is an inherent error in the correction due to the finite width
# of the detectors pixels.  The inside and outside edges of the pixel
# will correct to slightly different 2θ and ɸ values.  When this exceeds
# the target resolution there is no longer a gain from integrating
# to higher ɸ/z.

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from multihead.config import AnalyzerConfig
from multihead.corrections import tth_from_z

from hrd_tools.detector_stats import detectors


# %%
# Configuration and setup for plotting
cfg = AnalyzerConfig(
    910,
    120,
    np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
    2 * np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
    detector_roll=0,
)

# Use all available detectors
selected_detectors = list(detectors.keys())

# Calculate detector properties (all in mm)
det_props = {}
for name in selected_detectors:
    det = detectors[name]
    width = det.sensor_shape[1] * det.pixel_pitch * 1e-3  # convert µm to mm
    pixel_size = det.pixel_pitch * 1e-3  # convert µm to mm
    det_props[name] = {"width": width, "pixel_size": pixel_size}
print(det_props)

# %%
# Plot 1: Error vs z position for different 2theta values
# Group detectors by pixel size
pixel_size_groups = {}
for det_name in selected_detectors:
    pixel_size = det_props[det_name]["pixel_size"]
    if pixel_size not in pixel_size_groups:
        pixel_size_groups[pixel_size] = []
    pixel_size_groups[pixel_size].append(det_name)

# Sort pixel sizes for consistent ordering
sorted_pixel_sizes = sorted(pixel_size_groups.keys())

# Create subplots - one per pixel size
n_pixel_sizes = len(sorted_pixel_sizes)
fig1, axes1 = plt.subplots(
    1,
    n_pixel_sizes,
    layout="constrained",
    figsize=(4 * n_pixel_sizes, 4),
    dpi=100,
    sharey=True,
    sharex=True,
)

# Handle case where there's only one pixel size
if n_pixel_sizes == 1:
    axes1 = [axes1]

tth_arm_values = [10, 30, 60, 90]  # degrees
tth_colors = mpl.colormaps["viridis"](np.linspace(0, 1, len(tth_arm_values)))

# Find the largest detector width to set the plotting range (in mm)
max_width = max(det_props[det]["width"] for det in selected_detectors)

for subplot_idx, pixel_size in enumerate(sorted_pixel_sizes):
    ax = axes1[subplot_idx]

    # Create z positions at pixel edges from center to edge of detector (in mm)
    z_positions = np.arange(0, max_width / 2 + pixel_size, pixel_size)

    # Calculate correction for all tth values and positions at once
    (tth_all), (phi_all) = tth_from_z(
        z_positions.reshape(1, -1),
        np.array(tth_arm_values).reshape(-1, 1),
        cfg,
    )

    # Calculate the difference between adjacent positions (pixel edge effect)
    delta_2theta = np.abs(tth_all[:, 1:] - tth_all[:, :-1])

    # Z positions corresponding to pixel centers (in mm)
    z_pos = z_positions[:-1] + pixel_size / 2

    # Plot lines for different scatter 2theta values
    for i, tth in enumerate(tth_arm_values):
        ax.plot(
            z_pos,
            delta_2theta[i],
            color=tth_colors[i],
            label=f"$2\\Theta$ = {tth}°",
            linewidth=2,
        )

    # Create blended transform: data coords for x, axes coords for y
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # Add vertical lines for detectors with this pixel size
    for det_name in pixel_size_groups[pixel_size]:
        det_width = det_props[det_name]["width"]
        ax.axvline(x=det_width / 2, color="gray", linestyle="--", alpha=0.7)

        # Add detector name annotation at top of axes
        ax.annotate(
            det_name,
            xy=(det_width / 2, 1.0),
            xytext=(2, -2),
            textcoords="offset points",
            xycoords=trans,
            color="gray",
            fontsize=8,
            rotation=90,
            va="top",
            ha="left",
        )

    ax.set_title(f"{pixel_size * 1e3:.0f} µm pixel")
    ax.grid(True, alpha=0.3)

    # Only add legend to first subplot
    if subplot_idx == 0:
        ax.legend()
        ax.set_ylabel("2θ correction error (deg)")

    ax.set_xlabel("Distance from detector center (mm)")

fig1.suptitle("Inherent 2θ correction error vs 2Θ angle")
plt.show()

