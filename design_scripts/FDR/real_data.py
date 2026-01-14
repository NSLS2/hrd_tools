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
from multihead.raw_proc import get_roi_sum

# %%
roi_data = StringIO("""rois:
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

rois = DetectorROIs.from_yaml(roi_data)

# %%

root = Path("/mnt/store/bnl/cache/hrd/beamtime_cleanup/reorg/raw/v3/")
# the full data set
raw = open_data(root / "11bmb_2387_mda_defROI", 3)
detector_number = 1
data = raw.get_detector(detector_number)
# get the ROI subsection
rslc, cslc = rois.rois[1].to_slices()
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
cmap = mpl.colormaps["plasma"].with_extremes(under="w")

norm = mpl.colors.Normalize(1, 20)
for j, ax in enumerate(axa.flat):
    offset = j - 15
    frame = center_frame + offset
    ax.set_title(f"Δ={offset}", size="small")
    ax.xaxis.set_major_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.NullLocator())
    im = ax.imshow(data[frame].todense(), cmap=cmap, norm=norm, origin="lower")
fig.colorbar(im, ax=axa[-1, :], shrink=0.6, location="bottom")
# fig.savefig("/tmp/through_peak.png", dpi=300)

# %%

fig, axd = plt.subplot_mosaic(
    "A;B", layout="constrained", figsize=(3.5, 3.5), height_ratios=(5, 1), sharex=True
)
axd["A"].plot(point_eqiv / point_eqiv.max(), label="full detector", ls="--")
axd["A"].plot(roi_sum / roi_sum.max(), label="roi", alpha=0.75)
axd["B"].set_xlabel("frame number")
axd["A"].set_ylabel("normalized I (arb)")
axd["A"].legend()

axd["B"].plot(
    point_eqiv / point_eqiv.max() - roi_sum / roi_sum.max(), label="point - roi"
)
axd["B"].set_ylim(top=0.02)

fig.savefig("/tmp/detector_sum.png", dpi=300)

# %%

fig_kyho = plt.figure(layout="constrained", figsize=(4, 5))
ax_d = fig_kyho.subplot_mosaic("AB", width_ratios=(1, 5), sharey=True)
im = ax_d["B"].imshow(
    khymo,
    aspect="auto",
    extent=(
        0,
        khymo.shape[1],
        arm_tth.min(),
        arm_tth.max(),
    ),
    origin="lower",
    vmin=1,
    cmap=cmap,
)
ax_d["A"].plot(roi_sum, arm_tth)
cb = fig.colorbar(im, ax=ax_d["B"], location="right")
ax_d["A"].xaxis.set_major_formatter(
    lambda x, pos: f"{int(x // 1000)}k" if x != 0 else "0"
)
ax_d["A"].set_xlabel("$I_{sum}$ (photons)")
ax_d["A"].set_ylabel(r"$2\Theta$ (deg)")
ax_d["B"].set_xlabel("pixel")
cb.set_label("Intensity (photons)")
fig_kyho.savefig("/tmp/khymo_full.png", dpi=300)
ax_d["A"].set_ylim(6.42, 6.48)
fig_kyho.set_size_inches(3.9, 2.9)
fig_kyho.savefig("/tmp/khymo_zoom.png", dpi=300)
# %%

cosmic_index = 13213
fig, ax = plt.subplots(layout="constrained")
ax.imshow(data[cosmic_index].todense(), cmap=cmap, norm=norm, origin="lower")
