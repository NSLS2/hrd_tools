"""Divergence in intercept geometry when ignoring the gauge volume's extent."""

# %%
import matplotlib.pyplot as plt
import numpy as np

import _fdr_params

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)


def intercept(tth, w, d):
    tth = np.deg2rad(tth)
    w = np.deg2rad(w)
    return np.sin(np.pi - w) * d / np.sin(w + tth - np.pi)


d = 2                                          # mm
tth = np.linspace(-10, 120, 1024)              # deg

# %% [markdown]
#
# If we do not consider the extended nature of the gauge
# volume we get divergences

# %%
fig, ax = plt.subplots(layout="constrained")

for w in np.linspace(1, 179, 5):
    ax.plot(tth, intercept(tth, w, d), label=f"{w=}")
ax.set_ylim(-1.5 * d, 1.5 * d)
ax.legend()
_save(fig, "off_center.png")
_fdr_params.maybe_show(_args)
