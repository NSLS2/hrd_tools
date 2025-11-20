# %% [markdown]
# # Maximum roll of diffractometer
#
# If the detector is off vertical relative to the electron beam orbit
# we will mix the horizontal and vertical diveregences
#

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar

# %%
v_div = 0.9
h_div = 18.0
# accept 1% increase in divergence
max_div_perc = 0.005
max_div = v_div * (1 + max_div_perc)


# %%
def div(phi):
    phi = np.deg2rad(phi)
    return np.cos(phi) * v_div + np.sin(phi) * h_div


def max_phi_root(phi, target):
    return div(phi) - target


# %%

max_phi = root_scalar(max_phi_root, args=(max_div,), bracket=[0.0001, 0.1])

# %%
fig, ax = plt.subplots(layout="constrained")

phi = np.linspace(0, 0.05, 1028)

# TODO, plotting may be better as div/v_div
ax.plot(phi, div(phi))
ax.axhline(max_div, color="k", linestyle="--")
ax.axvline(max_phi.root, color="k", linestyle="--")

ax.set_ylabel("effective divergence (mrad)")
ax.set_xlabel("diffractometer roll (deg)")
ax.set_title(f"maximum of {100 * max_div_perc}% divergence degradation")
ax.annotate(
    f"""{v_div=:.2g} mrad
{h_div=:.1f} mrad
""",
    xy=(0, 1),
    xycoords="axes fraction",
    xytext=(5, -5),
    textcoords="offset points",
    va="top",
)
ax.annotate(
    f"$dff_{{roll}}$={max_phi.root * 1000:.2f} mdeg",
    xy=(max_phi.root, max_div),
    xytext=(5, -5),
    textcoords="offset points",
    va="top",
)
