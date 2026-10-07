# %% [markdown]
# The goal here is to generate a set of figures demonstrating what
# the raw data looks like and how we transform it to the 1D scattering
# curves.
#
# %%
from pathlib import Path

import _fdr_params
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import numpy.typing as npt
import scipy.signal
import tqdm
from matplotlib.animation import PillowWriter
from matplotlib.patches import Rectangle
from multihead.config import AnalyzerConfig, BankCalibration, DetectorROIs, SpectraCalib
from multihead.file_io import HRDRawBase, open_data
from multihead.raw_proc import scale_tth

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)

# %%
rois = DetectorROIs.from_yaml(_fdr_params.rois_path())

# %%
_rd = _fdr_params.real_data()
root = Path(_rd["root"])
raw = open_data(root / _rd["dataset"], _rd["version"])
detector_number = 1
data = raw.get_detector(detector_number)
# get the ROI subsection
rslc, cslc = rois.rois[detector_number].to_slices()
roi_data = data[:, rslc, cslc]
# treat detector as point detector
point_eqiv = data.sum(axis=(2), dtype=np.uint32).todense().sum(axis=1)
roi_sum = roi_data.sum(axis=(2), dtype=np.uint32).todense().sum(axis=1)
khymo = roi_data.sum(axis=1, dtype=np.uint32).todense()
# arm data
arm_tth = raw.get_arm_tth()

# %%
locs, props = scipy.signal.find_peaks(
    point_eqiv, width=(None, None), height=(None, None)
)
indx = np.argsort(props["peak_heights"])[:-5:-1]
peak_locations = locs[indx]
peak_heights = props["peak_heights"][indx]
peak_widths = props["widths"][indx]

# %%
fig, axa = plt.subplots(5, 5, layout="constrained", figsize=(3.75, 5))
center_frame = peak_locations[1]
cmap = mpl.colormaps["plasma"].with_extremes(under="w", over="r")

norm = mpl.colors.Normalize(1, 30)
for j, ax in enumerate(axa.flat):
    offset = j - 15
    frame = center_frame + offset
    if offset != 0:
        ax.set_title(rf"$\Delta_{{fr}}$={offset}", size="small")
    else:
        ax.set_title(rf"$2\Theta$={arm_tth[frame]:.3f}°", size="small")
    ax.xaxis.set_major_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.NullLocator())
    im = ax.imshow(data[frame].todense(), cmap=cmap, norm=norm, origin="lower")
cb = fig.colorbar(im, ax=axa[-1, 3:], location="bottom", extend="min")
cb.set_ticks([1, 10, 20, 30])
cb.set_label("photons", loc="right")
fig.text(
    0,
    0.02,
    rf"$\Delta 2\Theta$ = {np.round(np.mean(np.diff(arm_tth)) * 1000, 2):.1g}mdeg",
    ha="left",
    in_layout=True,
)
_save(fig, "through_peak.png")

# %%
fig, axd = plt.subplot_mosaic(
    "A;B", layout="constrained", figsize=(3.5, 3.5), height_ratios=(5, 1), sharex=True
)
axd["A"].plot(arm_tth, point_eqiv / point_eqiv.max(), label="full detector", ls="--")
axd["A"].plot(arm_tth, roi_sum / roi_sum.max(), label="roi", alpha=0.75)
axd["B"].set_xlabel(r"$2\Theta$ (deg)")
axd["A"].set_ylabel("normalized I (arb)")
axd["A"].legend()

axd["B"].plot(
    arm_tth,
    point_eqiv / point_eqiv.max() - roi_sum / roi_sum.max(),
    label="point - roi",
)
axd["B"].set_ylim(top=0.02)
axd["B"].set_xlim(right=30)

_save(fig, "detector_sum.png")

# %%
cmap = mpl.colormaps["plasma"].with_extremes(under="w", over="r")
norm = mpl.colors.Normalize(1, 30)
fig_all = plt.figure(layout="constrained", figsize=(7.5, 5))
fig_kymo, fig_insets = fig_all.subfigures(1, 2)
fig_zoomA, fig_zoomB = fig_insets.subfigures(2, 1)
ax_d_main = fig_kymo.subplot_mosaic("BA", width_ratios=(5, 1), sharey=True)
ax_d_zoomA = fig_zoomA.subplot_mosaic("BA", width_ratios=(5, 1), sharey=True)
ax_d_zoomB = fig_zoomB.subplot_mosaic("BA", width_ratios=(5, 1), sharey=True)


def plot_with_parasite(ax_I, ax_K, cmap):
    im = ax_K.imshow(
        khymo,
        aspect="auto",
        extent=(0, khymo.shape[1], arm_tth.min(), arm_tth.max()),
        origin="lower",
        vmin=0.5,
        vmax=550,
        cmap=cmap,
    )
    ax_I.plot(roi_sum, arm_tth)

    ax_I.xaxis.set_major_formatter(
        lambda x, pos: f"{int(x // 1000)}k" if x != 0 else "0"
    )
    ax_I.set_xlabel("$I_{sum}$ (#)")
    ax_K.set_ylabel(r"$2\Theta$ (deg)")
    ax_K.set_xlabel("pixel")
    return im


im = plot_with_parasite(ax_d_main["A"], ax_d_main["B"], cmap)
plot_with_parasite(ax_d_zoomA["A"], ax_d_zoomA["B"], cmap)
plot_with_parasite(ax_d_zoomB["A"], ax_d_zoomB["B"], cmap)

delta = 0.06
startA = 6.42
startB = 24.32
ax_d_zoomA["A"].set_ylim(startA, startA + delta)
ax_d_zoomB["A"].set_ylim(startB, startB + delta)

cb = fig_all.colorbar(
    im, ax=[ax_d_main["B"]], location="left", aspect=50, extend="min", extendfrac=0.02
)
_save(fig_all, "khymo_full.png")

# %%
cosmic_index = 13213
cosmic_data = raw.get_detector(1)[cosmic_index].todense()

norm = mpl.colors.BoundaryNorm(np.array([1, 2, 3]) - 0.5, 2)
cmap = mpl.colormaps["plasma"].with_extremes(under="w").resampled(2)
fig, ax = plt.subplots(layout="constrained", figsize=(3.9, 2.9))
im = ax.imshow(cosmic_data, cmap=cmap, norm=norm, origin="lower")
cosmic_roi = rois.rois[1]

rect = Rectangle(
    (cosmic_roi.cslc.start, cosmic_roi.rslc.start),
    cosmic_roi.cslc.stop - cosmic_roi.cslc.start,
    cosmic_roi.rslc.stop - cosmic_roi.rslc.start,
    linewidth=2,
    edgecolor="gray",
    facecolor="none",
    alpha=0.7,
)
ax.add_artist(rect)
ax.annotate(
    "ROI",
    xy=(1, 1),
    xycoords=rect,
    xytext=(0, 5),
    textcoords="offset points",
    va="bottom",
    ha="right",
    color=rect.get_edgecolor(),
)
ax.annotate(
    "Cosmic Ray",
    xy=(50, 190),
    xytext=(30, 5),
    textcoords="offset points",
    va="center",
    ha="left",
    arrowprops=dict(arrowstyle="->"),
    color="k",
)
cbar = fig.colorbar(im, extend="min")
cbar.set_ticks([1, 2])
_save(fig, "cosmic.png")

# %%


def estimate_crystal_offsets(
    raw: HRDRawBase,
    flats: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]],
) -> dict[int, float]:
    bin_size = raw.get_nominal_bin()
    out: dict[int, float] = {}
    iterator = iter(flats.items())
    det, (_, ref) = next(iterator)
    (Npts,) = ref.shape
    out[det] = cum_offset = 0.0

    for det, (_, I) in iterator:
        offset = np.argmax(np.correlate(ref, I, mode="full")) - Npts
        cum_offset += offset * bin_size
        out[det] = cum_offset
        ref = I

    return out


# %%
# Compute integrated (flat) spectra for all detectors using their ROIs
flats: dict[int, tuple[npt.NDArray[np.floating], npt.NDArray[np.uint16]]] = {}
for det, roi in tqdm.tqdm(rois.rois.items(), desc="integrating detectors"):
    det_data = raw.get_detector(det)
    rslc, cslc = roi.to_slices()
    I = det_data[:, rslc, cslc].sum(axis=1, dtype=np.uint32).todense().sum(axis=1)
    flats[det] = (arm_tth, I)

# %%
offsets = estimate_crystal_offsets(raw, flats)

default_wavelength = 0.8272  # Å for 15 keV
default_scale = 1.0

calibrations = {
    det: SpectraCalib(
        offset=offset,
        scale=default_scale,
        wavelength=default_wavelength,
        analyzer=AnalyzerConfig(
            R=910.0,
            Rd=120.0,
            center=128,
            theta_i=np.rad2deg(np.arcsin(default_wavelength / (2 * 3.1355))),
            theta_d=2 * np.rad2deg(np.arcsin(default_wavelength / (2 * 3.1355))),
            crystal_roll=0.0,
            crystal_yaw=0.0,
        ),
    )
    for det, offset in offsets.items()
}

calibration_config = BankCalibration(
    calibrations=calibrations,
    software={"name": "multihead", "version": "dev", "script": "real_data.py"},
    parameters={
        "num_detectors": len(offsets),
        "estimation_method": "correlation_based",
        "default_wavelength_nm": default_wavelength,
        "default_scale": default_scale,
    },
    pixel_pitch=0.055,
)
calibs = calibration_config.calibrations

# %%
mon = raw.get_monitor()

fig_spectra, ax_spectra = plt.subplots(layout="constrained", figsize=(7, 4))
for d, (tth, I) in flats.items():
    ax_spectra.plot(
        scale_tth(
            tth + calibs[d].offset,
            calibs[d].wavelength,
            calibration_config.average_wavelength,
        ),
        (I / mon) * calibs[d].scale,
        label=str(d),
    )
ax_spectra.legend(ncols=2, fontsize="small")
ax_spectra.set_xlabel(r"$2\theta$ (deg)")
ax_spectra.set_ylabel("I / monitor (arb)")
ax_spectra.set_title("Offset-corrected spectra — all detectors")
_save(fig_spectra, "aligned_spectra.png")

# %%
# Animated GIF cutting through the peak (same range as the 5x5 grid figure)
cmap_gif = mpl.colormaps["plasma"].with_extremes(under="w", over="r")
norm_gif = mpl.colors.Normalize(1, 30)
peak_tth = arm_tth[center_frame]

fig_gif, ax_gif = plt.subplots(layout="constrained", figsize=(4, 3.5))
im_gif = ax_gif.imshow(
    data[center_frame - 15].todense(), cmap=cmap_gif, norm=norm_gif, origin="lower"
)
cb_gif = fig_gif.colorbar(im_gif, extend="min")
cb_gif.set_ticks([1, 10, 20, 30])
cb_gif.set_label("photons")
ax_gif.xaxis.set_major_locator(mticker.NullLocator())
ax_gif.yaxis.set_major_locator(mticker.NullLocator())

txt_angle = ax_gif.text(
    0.02,
    0.98,
    "",
    transform=ax_gif.transAxes,
    va="top",
    ha="left",
    fontsize="small",
    color="w",
    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
)

_gif_out = _args.outdir / "through_peak.gif"
_writer = PillowWriter(fps=10)
with _writer.saving(fig_gif, _gif_out, dpi=_args.dpi):
    for j in range(25):
        offset = j - 15
        frame = center_frame + offset
        im_gif.set_data(data[frame].todense())
        delta_deg = (arm_tth[frame] - peak_tth) * 1000  # mdeg
        txt_angle.set_text(
            f"$2\\Theta$={arm_tth[frame]:.3f}°\n"
            f"$\\Delta 2\\Theta$={delta_deg:> 5.1f} mdeg\n"
            f"$\\Delta_{{fr}}$={offset}"
        )
        _writer.grab_frame()

# %%
# Single-frame figure: central frame only, same colormap/scale, no colorbar
fig_cf, ax_cf = plt.subplots(layout="constrained", figsize=(4, 3.5))
ax_cf.imshow(data[center_frame].todense(), cmap=cmap_gif, norm=norm_gif, origin="lower")
ax_cf.xaxis.set_major_locator(mticker.NullLocator())
ax_cf.yaxis.set_major_locator(mticker.NullLocator())
_save(fig_cf, "center_frame.png")

# %%
# Merge all 12 offset-corrected spectra into one.
# Strategy: interpolate raw counts and monitor onto a common tth grid
# (full union range), sum raw counts across detectors, then divide by
# total monitor.  This weights each point by monitor exposure.

step = np.mean(np.diff(arm_tth))

aligned_raw: dict[int, tuple[npt.NDArray, npt.NDArray]] = {}
tth_mins: list[float] = []
tth_maxs: list[float] = []
for d, (tth, I) in flats.items():
    aligned_tth = scale_tth(
        tth + calibs[d].offset,
        calibs[d].wavelength,
        calibration_config.average_wavelength,
    )
    # I is raw counts (not monitor-normalized); scale is a per-crystal
    # relative-efficiency correction so still apply it to the counts.
    aligned_raw[d] = (aligned_tth, I * calibs[d].scale)
    tth_mins.append(float(aligned_tth.min()))
    tth_maxs.append(float(aligned_tth.max()))

common_tth = np.arange(min(tth_mins), max(tth_maxs), step)

merged_counts = np.zeros_like(common_tth)
merged_mon = np.zeros_like(common_tth)

for d, (tth_a, I_a) in aligned_raw.items():
    interp_I = np.interp(common_tth, tth_a, I_a, left=np.nan, right=np.nan)
    interp_mon = np.interp(common_tth, tth_a, mon, left=np.nan, right=np.nan)
    valid = np.isfinite(interp_I)
    merged_counts[valid] += interp_I[valid]
    merged_mon[valid] += interp_mon[valid]

merged_I = np.divide(
    merged_counts, merged_mon, out=np.zeros_like(merged_counts), where=merged_mon > 0
)

# %%
fig_merged, ax_merged = plt.subplots(layout="constrained", figsize=(7, 4))
ax_merged.plot(common_tth, merged_I, color="k", lw=0.5)
ax_merged.set_xlabel(r"$2\theta$ (deg)")
ax_merged.set_ylabel("I / monitor (arb)")
ax_merged.set_title(f"Merged spectrum \u2014 {len(aligned_raw)} detectors")
_save(fig_merged, "merged_spectrum.png")

_fdr_params.maybe_show(_args)
