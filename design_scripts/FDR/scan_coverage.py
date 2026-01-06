# %% [markdown]
# # Scatter coverage as a function of arm angle
#

# %%
import dataclasses

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from multihead.config import AnalyzerConfig
from multihead.corrections import tth_from_z
from scipy import optimize
from matplotlib.colors import Normalize

from hrd_tools.xrt import CrystalProperties
from hrd_tools.detector_stats import detectors
from tqdm import tqdm
import concurrent.futures

# %%
mpl.rcParams["savefig.dpi"] = 300


# %%

props = CrystalProperties.create(E=40)

# Configure analyzer with realistic parameters
cfg = AnalyzerConfig(
    910,  # R: sample to crystal distance (mm)
    120,  # Rd: crystal to detector distance (mm)
    props.bragg_angle,
    2 * props.bragg_angle,
    detector_roll=0,
)

# %%

det = detectors["medipix3"]

# %%
#
#
# bins = np.arange(12, 45, 1e-4)
# vals = np.zeros(len(bins) - 1)
# arm_pos = np.arange(12, 14.5, 1e-4)
# z = (
#     (np.arange(0, det.sensor_shape[1]) - (det.sensor_shape[1] / 2))
#     * det.pixel_pitch
#     / 1_000
# )
#
# fig, ax = plt.subplots(1, 1, layout="constrained", sharey=True)
# for offset in tqdm([0, 2, 4, 6, 8, 10, 12]):
#     corrected_tths, _ = tth_from_z(
#         z.reshape(-1, 1), arm_pos.reshape(1, -1) + offset, cfg
#     )
#
#     crystal_vals, _ = np.histogram(corrected_tths.ravel(), bins=bins)
#     vals += crystal_vals
#
#     # ax.stairs(crystal_vals, bins)
#     ax.plot(bins[:-1], crystal_vals)
# # ax.stairs(vals, bins)
# ax.plot(bins[:-1], vals)
#
# %%
scan_range = 8
bins = np.arange(0, 90, 1e-4)
vals = np.zeros(len(bins) - 1)
arm_pos = np.arange(0, scan_range, 1e-4).reshape(1, -1)
z = (
    (np.arange(0, det.sensor_shape[1]) - (det.sensor_shape[1] / 2))
    * det.pixel_pitch
    / 1_000
).reshape(-1, 1)
offsets = [j * 2 + k * (11 * 2 + 6) for k in range(4) for j in range(12)]
# currently the correctino code can't go past 90deg
offsets = [_ for _ in offsets if _ < 90 - scan_range]

norm = Normalize(min(offsets), max(offsets))
cmap = mpl.colormaps["viridis"]
fig, ax = plt.subplots(1, 1, layout="constrained", sharey=True)


def compute_bins(z, arm_pos, offset, bins, cfg):
    corrected_tths, _ = tth_from_z(z, arm_pos + offset, cfg)

    crystal_vals, _ = np.histogram(corrected_tths.ravel(), bins=bins)

    return crystal_vals


# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {
        executor.submit(compute_bins, z, arm_pos, offset, bins, cfg): offset
        for offset in offsets
    }
    for future in tqdm(
        concurrent.futures.as_completed(future_to_url), total=len(future_to_url)
    ):
        offset = future_to_url[future]
        try:
            crystal_vals = future.result()
        except Exception as exc:
            print("%r generated an exception: %s" % (offset, exc))
        else:
            ax.plot(bins[:-1], crystal_vals, color=cmap(norm(offset)))
            vals += crystal_vals

    ax.plot(bins[:-1], vals, lw=5, alpha=0.5)
