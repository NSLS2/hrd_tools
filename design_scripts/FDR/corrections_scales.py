# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
# # Corrections Analysis: Arm vs Scattering Angle Differences
#
# This notebook demonstrates the geometric corrections needed for analyzing scattering data
# from a multi-head diffractometer. It shows two key relationships:
#
# 1. How the arm angle differs from the nominal scattering angle for various scattering angles
# 2. How the corrected scattering angle differs from the arm angle for various arm positions
#
# These corrections account for the finite size of the detector and the geometry of the
# analyzer crystals in the diffractometer setup.

# %% [markdown]
# ## Setup and Configuration
#
# Import required libraries and configure the analyzer with realistic parameters.

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from multihead.config import AnalyzerConfig
from multihead.corrections import arm_from_z, tth_from_z

from hrd_tools.detector_stats import detectors

from hrd_tools.xrt import CrystalProperties
# %%

mpl.rcParams['savefig.dpi'] = 300

# %%
props = CrystalProperties.create(E=40)

# Configure analyzer with realistic parameters
cfg = AnalyzerConfig(
    910,  # R: sample to crystal distance (mm)
    120,  # Rd: crystal to detector distance (mm)
    props.bragg_angle,
    2*props.bragg_angle,
    detector_roll=0,
)
# in mm
det_size = dict(
    sorted(
        {
            k: det.pixel_pitch * det.sensor_shape[1] / 1000
            for k, det in detectors.items()
        }.items(),
        key=lambda x: x[1],
    )
)


# Define detector positions along the axial direction
z = np.linspace(-20, 20, 256)


# %% [markdown]
# ## Arm Angle vs Scattering Angle Difference
#
# This plot shows how the arm angle (2Θ) differs from the scattering angle (2θ)
# as a function of axial position on the detector for various scattering angles.
# The difference accounts for the geometric corrections due the missalignment of the analyzer
# crystals and the finite size of the detector.
#
# This is useful to understand the effect of the analyzer crystal with area detector
# passing through the Debye-Scherrer cone.  This is not useful for data reduction.
#
# Shown as the difference so everything fits on the same axis.

# %%
fig, ax = plt.subplots(layout="constrained")
thetas = np.arange(5, 90, 10)[::-1]
cmap = mpl.colormaps["viridis"]

arm_tths, _ = arm_from_z(z.reshape(1, -1), thetas.reshape(-1, 1), cfg)

for arm_tth, tth, color in zip(
    arm_tths, thetas, cmap(np.linspace(0, 1, len(thetas))), strict=True
):
    ax.plot(z, arm_tth - tth, label=rf"$2\theta = {tth:g}°$", color=color)

ax.legend()
ax.set_ylabel(r"$2\Theta - 2\theta$ (deg)")
ax.set_xlabel(r"axial offset from center of detector (mm)")
ax.set_title("Arm Angle Correction given Scattering Angle")

side = 1
for det_name, det_size_val in det_size.items():
    x_pos = side * det_size_val / 2
    if x_pos > 20:
        continue
    ax.axvline(
        x_pos,
        color="gray",
        linestyle="--",
        alpha=0.5,
    )
    # Add text label using BlendedTransform
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        x_pos,
        0.05,
        det_name,
        transform=trans,
        rotation=90,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=8,
        alpha=0.7,
    )
    side *= -1
ax.set_ylim(top=.005, bottom=-0.04)
ax.set_xlim(-20, 20)
plt.show(block=False)

# %% [markdown]
# ## Corrected Scattering Angle vs Arm Angle Difference
#
# This plot shows how the corrected scattering angle (2θ) differs from the arm angle (2Θ)
# as a function of axial position on the detector for various fixed arm angles.
#
# This represents the correction, needed for data reduction as we experimentally
# know the arm position and need to compute the true scattering angle.
# %%
fig, ax = plt.subplots(layout="constrained")
arm_angles = np.arange(5, 90, 10)[::-1]
cmap = mpl.colormaps["viridis"]

corrected_tths, _ = tth_from_z(z.reshape(1, -1), arm_angles.reshape(-1, 1), cfg)

for corr_tth, arm_tth, color in zip(
    corrected_tths, arm_angles, cmap(np.linspace(0, 1, len(arm_angles))), strict=True
):
    ax.plot(z, -corr_tth + arm_tth, label=rf"$2\Theta = {arm_tth:g}°$", color=color)

# Add detector boundaries and labels
side = 1
for det_name, det_size_val in det_size.items():
    x_pos = side * det_size_val / 2
    if x_pos > 20:
        continue
    ax.axvline(
        x_pos,
        color="gray",
        linestyle="--",
        alpha=0.5,
    )
    # Add text label using BlendedTransform
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        x_pos,
        0.05,
        det_name,
        transform=trans,
        rotation=90,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=8,
        alpha=0.7,
    )
    side *= -1

ax.legend()
ax.set_ylabel(r"$2\Theta - 2\theta$ (deg)")
ax.set_xlabel(r"axial offset from center of detector (mm)")
ax.set_title("Scattering Angle Correction given Arm Position")
ax.set_ylim(top=.005, bottom=-0.04)
ax.set_xlim(-20, 20)
plt.show(block=True)
