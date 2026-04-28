# %% [markdown]
#
# Compute the minimum crystal-sample distance given a gauge depth
#
# Based on where the ray from the down stream edge of the gauge volume of the leading
# crystal crosses the upstream edge of the trailing crystal.

"""Minimum crystal-to-sample distance vs gauge depth and arm pitch."""

# %%
import matplotlib.pyplot as plt
import numpy as np

import _fdr_params

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)
_layout = _fdr_params.layout()


# %%
def crossing_r(tth, delta, gague_depth):
    return gague_depth * np.sin(np.deg2rad(tth - delta)) / np.sin(np.deg2rad(delta))


def distance_to_seperation(delta, target_width):
    return target_width / (2 * np.sin(np.deg2rad(delta / 2)))


# %%

tth = np.linspace(0, 90, 128)                  # deg
gague_depth = 5                                # mm   (default; sweep below)
delta = _layout["pitch_deg"]                   # deg  (between adjacent crystals)
clearance = 25                                 # mm

# %%

fig, ax = plt.subplots(layout="constrained")
for gv in [3, 5, 8]:
    (ln,) = ax.plot(
        tth, crossing_r(tth, delta, gv), label=f"crossing for {gv=}mm", linestyle="--"
    )

    ax.plot(
        tth,
        crossing_r(tth, delta, gv) + distance_to_seperation(delta, clearance),
        label=f"with clearance for {gv=}mm",
        linestyle="-",
        color=ln.get_color(),
    )
ax.set_ylabel("distance from sample (mm)")
ax.set_xlabel(r"$2\Theta$ (deg)")
ax.set_title(rf"$\delta={delta}\degree$, {clearance=}mm")
ax.legend()
_save(fig, "min_crystal_distance.png")
_fdr_params.maybe_show(_args)
