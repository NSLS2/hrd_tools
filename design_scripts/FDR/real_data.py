# %% [markdown]
# The goal here is to generate a set of figures demonstrating what
# the raw data looks like and how we transform it to the 1D scattering
# curves.
#
# %%
from io import StringIO
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.signal
from multihead.config import DetectorROIs
from multihead.file_io import open_data


# %%
roi_data_str = StringIO("""rois:
- detector_number: 1
  roi_bounds:
    cslc:
    - 0
    - 208
    rslc:
    - 120
    - 136
- detector_number: 2
  roi_bounds:
    cslc:
    - 25
    - 240
    rslc:
    - 77
    - 94
- detector_number: 3
  roi_bounds:
    cslc:
    - 11
    - 219
    rslc:
    - 176
    - 194
- detector_number: 4
  roi_bounds:
    cslc:
    - 25
    - 236
    rslc:
    - 188
    - 204
- detector_number: 5
  roi_bounds:
    cslc:
    - 12
    - 221
    rslc:
    - 185
    - 202
- detector_number: 6
  roi_bounds:
    cslc:
    - 30
    - 240
    rslc:
    - 122
    - 139
- detector_number: 7
  roi_bounds:
    cslc:
    - 15
    - 220
    rslc:
    - 162
    - 179
- detector_number: 8
  roi_bounds:
    cslc:
    - 30
    - 238
    rslc:
    - 70
    - 87
- detector_number: 9
  roi_bounds:
    cslc:
    - 13
    - 217
    rslc:
    - 166
    - 183
- detector_number: 10
  roi_bounds:
    cslc:
    - 32
    - 238
    rslc:
    - 115
    - 132
- detector_number: 11
  roi_bounds:
    cslc:
    - 0
    - 211
    rslc:
    - 122
    - 139
- detector_number: 12
  roi_bounds:
    cslc:
    - 36
    - 237
    rslc:
    - 59
    - 78""")

rois = DetectorROIs.from_yaml(roi_data_str)

# %%

root = Path("/mnt/store/bnl/cache/hrd/beamtime_cleanup/reorg/raw/v3/")
save_base = Path("/home/tcaswell/windows_bridge/")
# the full data set
raw = open_data(root / "11bmb_2387_mda_defROI", 3)
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
# peak with lowest angle shows most dramatic effect
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
fig.savefig(save_base / "through_peak.png", dpi=300)

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

fig.savefig(save_base / "detector_sum.png", dpi=300)
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
    extent=(
        0,
        khymo.shape[1],
        arm_tth.min(),
        arm_tth.max(),
    ),
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

plot_with_parasite(ax_d_main['A'], ax_d_main['B'], cmap)
plot_with_parasite(ax_d_zoomA['A'], ax_d_zoomA['B'], cmap)
plot_with_parasite(ax_d_zoomB['A'], ax_d_zoomB['B'], cmap)

delta = 0.06
startA = 6.42
startB = 24.32
ax_d_zoomA["A"].set_ylim(startA, startA+delta)
ax_d_zoomB["A"].set_ylim(startB, startB+delta)

cb = fig_all.colorbar(im, ax=[ax_d_main["B"]], location="left", aspect=50, extend='min', extendfrac=.02)
#cb.set_label("Intensity (photons)")
fig_all.savefig(save_base / "khymo_full.png", dpi=300)

# %%

from matplotlib.patches import Rectangle
cosmic_index = 13213
cosmic_data = raw.get_detector(1)[cosmic_index].todense()


norm = mpl.colors.BoundaryNorm(np.array([1, 2, 3]) - .5, 2)
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
an2 = ax.annotate("ROI",
                  xy=(1, 1), xycoords=rect,  # (1, 0.5) of an1's bbox
                  xytext=(0, 5), textcoords="offset points",
                  va="bottom", ha="right",
                  #bbox=dict(boxstyle="round", fc="w"),
                  #arrowprops=dict(arrowstyle="->"),
                  color=rect.get_edgecolor(),
                  # alpha=rect.get_alpha()
                  )
an2 = ax.annotate("Cosmic Ray",
                  xy=(50, 190),
                  xytext=(30, 5), textcoords="offset points",
                  va="center", ha="left",
                  #bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->"),
                  color='k',
                  #alpha=rect.get_alpha()
                  )
cbar = fig.colorbar(im, extend='min')
cbar.set_ticks([1, 2])
fig.savefig(save_base / "cosmic.png", dpi=300)
