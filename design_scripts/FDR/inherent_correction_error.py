# %% [markdown]
# # Ineherent correction uncertainty
#
# There is an inherent uncertainty in the correction due to the finite width
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

import _fdr_params
from hrd_tools.detector_stats import detectors
from hrd_tools.xrt import CrystalProperties

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)
_blessed = _fdr_params.complete_config()
_e_keV = _args.energy_keV if _args.energy_keV is not None else _blessed.source.E_incident / 1000.0
_props = CrystalProperties.create(E=_e_keV)

# %%
# Configuration and setup for plotting
cfg = _fdr_params.analyzer_multihead(energy_keV=_args.energy_keV)

# Use all available detectors
selected_detectors = list(detectors.keys())

# Calculate detector properties (all in mm)
det_props = {}
for name in selected_detectors:
    det = detectors[name]
    width = det.sensor_shape[1] * det.pixel_pitch * 1e-3      # mm
    pixel_size = det.pixel_pitch * 1e-3                       # mm
    det_props[name] = {"width": width, "pixel_size": pixel_size}
print(det_props)

# %%
# Plot 1: Uncertainty vs z position for different 2theta values
pixel_size_groups = {}
for det_name in selected_detectors:
    pixel_size = det_props[det_name]["pixel_size"]
    pixel_size_groups.setdefault(pixel_size, []).append(det_name)

sorted_pixel_sizes = sorted(pixel_size_groups.keys())

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

if n_pixel_sizes == 1:
    axes1 = [axes1]

tth_arm_values = [10, 20, 30, 60, 90]          # deg
tth_colors = mpl.colormaps["viridis"](np.linspace(0, 1, len(tth_arm_values)))

max_width = max(det_props[det]["width"] for det in selected_detectors)

for subplot_idx, pixel_size in enumerate(sorted_pixel_sizes):
    ax = axes1[subplot_idx]

    z_positions = np.arange(0, max_width / 2 + pixel_size, pixel_size)

    (tth_all), (phi_all) = tth_from_z(
        z_positions.reshape(1, -1),
        np.array(tth_arm_values).reshape(-1, 1),
        cfg,
    )

    delta_2theta = np.abs(tth_all[:, 1:] - tth_all[:, :-1])

    z_pos = z_positions[:-1] + pixel_size / 2

    for i, tth in enumerate(tth_arm_values):
        ax.plot(
            z_pos,
            1000 * delta_2theta[i],
            color=tth_colors[i],
            label=f"$2\\Theta$ = {tth}°",
            linewidth=2,
        )

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    trans2 = blended_transform_factory(ax.transAxes, ax.transData)
    widths_hit = {}
    for det_name in pixel_size_groups[pixel_size]:
        det_width = det_props[det_name]["width"]
        scale = widths_hit.get(det_width, 0)
        if scale == 0:
            ax.axvline(x=det_width / 2, color="gray", linestyle="--", alpha=0.7)

        ax.annotate(
            det_name,
            xy=(det_width / 2, 1.0 - 0.25 * scale),
            xytext=(2, -2),
            textcoords="offset points",
            xycoords=trans,
            color="gray",
            fontsize=8,
            rotation=90,
            va="top",
            ha="left",
        )
        widths_hit[det_width] = scale + 1

    ax.set_title(f"{pixel_size * 1e3:.0f} µm pixel")
    ax.grid(True, alpha=0.3)

    ax.axhline(1e-1, ls=":", alpha=0.5, color="k")
    ax.annotate(
        "Maximum\nuncertainty",
        xy=(0, 0.1),
        xycoords=trans2,
        xytext=(2, 2),
        textcoords="offset points",
    )
    if subplot_idx == 0:
        ax.legend()
        ax.set_ylabel("2θ correction uncertainty  (mdeg)")

    ax.set_xlabel("Distance from detector center (mm)")

fig1.suptitle("Inherent 2θ correction uncertainty vs 2Θ angle")
_save(fig1, "inherent_correction_error_per_detector.png")


# %%
# Figure 2: effect of crystal position on error
tth_arm_values_fig2 = np.linspace(1, 90, 90)   # deg

# Crystal-position sweep — intentionally varied (not a "blessed" parameter).
total_distance = _blessed.analyzer.R + _blessed.analyzer.Rd     # mm

cfgs_fig2 = [
    AnalyzerConfig(
        middle,                                                 # R (mm)
        total_distance - middle,                                # Rd (mm)
        _props.bragg_angle,
        2 * _props.bragg_angle,
        crystal_roll=1 / 1000,                                  # deg
    )
    for middle in [100, 250, 500, 900, 1000]
]

# Pixel size of the blessed baseline detector
pixel_size_fig2 = _fdr_params.detector().pixel_pitch / 1000.0   # mm

z_edges = np.array([0, pixel_size_fig2])

delta_2theta_all = []
for cfg in cfgs_fig2:
    (tth_corrected), (_phi) = tth_from_z(
        z_edges.reshape(1, -1),
        tth_arm_values_fig2.reshape(-1, 1),
        cfg,
    )
    delta_2theta = np.abs(tth_corrected[:, 1] - tth_corrected[:, 0])
    delta_2theta_all.append(delta_2theta)

delta_2theta_all = np.array(delta_2theta_all)

fig2, (ax_ptp, ax_raw) = plt.subplots(
    2, 1, layout="constrained", figsize=(4, 4.5), dpi=100, sharex=True
)

ax_ptp.plot(
    tth_arm_values_fig2,
    np.ptp(delta_2theta_all, axis=0),
    label=f"{pixel_size_fig2 * 1e3:.0f} µm pixel (ptp)",
    color="C0",
    lw=2,
)

for i, middle in enumerate([100, 250, 500, 900, 1000]):
    ax_raw.plot(
        tth_arm_values_fig2,
        delta_2theta_all[i],
        label=f"{middle} mm",
        alpha=0.6,
        lw=1,
    )

ax_ptp.legend(loc="upper left")
ax_raw.legend(loc="upper right")
ax_raw.set_xlabel(r"$2\theta$ (deg)")
ax_ptp.set_ylabel(r"$\Delta$(2θ error) (deg)")
ax_raw.set_ylabel(r"2θ error (deg)")

ax_ptp.set_xlim(0, 90)
ax_ptp.grid(True, alpha=0.3)
ax_raw.grid(True, alpha=0.3)

fig2.suptitle(
    "Peak-to-peak variation in 2θ correction error for crystal positions 100-1000 mm"
)

_save(fig2, "inherent_correction_error_crystal_position.png")
_fdr_params.maybe_show(_args)
