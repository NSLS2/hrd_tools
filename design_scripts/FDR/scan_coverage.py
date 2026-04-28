# %% [markdown]
# # Scatter coverage as a function of arm angle
#

# %%
import concurrent.futures

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from multihead.corrections import tth_from_z
from tqdm import tqdm

import _fdr_params

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)

mpl.rcParams["savefig.dpi"] = _args.dpi


# %%
cfg = _fdr_params.analyzer_multihead(energy_keV=_args.energy_keV)
det = _fdr_params.detector()
_layout = _fdr_params.layout()

# %%
scan_range = _layout["scan_range_deg"]
bins = np.arange(0, 90, 1e-4)
vals = np.zeros(len(bins) - 1)
arm_pos = np.arange(0, scan_range, 1e-4).reshape(1, -1)
z = (
    (np.arange(0, det.sensor_shape[1]) - (det.sensor_shape[1] / 2))
    * det.pixel_pitch
    / 1_000
).reshape(-1, 1)

# offsets[j, k] = j*pitch_deg  +  k*primary_offset_deg  for j in [0, crystals_per_bank)
# and k in [0, banks).  This matches Primary(Bank(N, pitch), banks, primary_offset).
_pitch = _layout["pitch_deg"]
_primary = _layout["primary_offset_deg"]
_n_cry = _layout["crystals_per_bank"]
_n_banks = _layout["banks"]
offsets = [j * _pitch + k * _primary for k in range(_n_banks) for j in range(_n_cry)]
# the correction code can't go past 90deg
offsets = [_ for _ in offsets if _ < 90 - scan_range]

norm = Normalize(min(offsets), max(offsets))
cmap = mpl.colormaps["viridis"]
fig, ax = plt.subplots(1, 1, layout="constrained", sharey=True)


def compute_bins(z, arm_pos, offset, bins, cfg):
    corrected_tths, _ = tth_from_z(z, arm_pos + offset, cfg)
    crystal_vals, _ = np.histogram(corrected_tths.ravel(), bins=bins)
    return crystal_vals


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    future_to_offset = {
        executor.submit(compute_bins, z, arm_pos, offset, bins, cfg): offset
        for offset in offsets
    }
    for future in tqdm(
        concurrent.futures.as_completed(future_to_offset), total=len(future_to_offset)
    ):
        offset = future_to_offset[future]
        try:
            crystal_vals = future.result()
        except Exception as exc:
            print(f"{offset!r} generated an exception: {exc}")
        else:
            ax.plot(bins[:-1], crystal_vals, color=cmap(norm(offset)))
            vals += crystal_vals

    ax.plot(bins[:-1], vals, lw=5, alpha=0.5)

ax.set_xlabel(r"$2\theta$ (deg)")
ax.set_ylabel("counts (arb.)")
ax.set_title(
    f"Scan coverage: {_n_banks} banks x {_n_cry} crystals, scan {scan_range}°"
)

_save(fig, "scan_coverage.png")
_fdr_params.maybe_show(_args)
