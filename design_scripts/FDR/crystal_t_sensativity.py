# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from hrd_tools.xrt import CrystalProperties

# %%
room = 295.0  # K
start, stop = room - 7, room + 7  #

E = 40  # kEv
steps = 128


# %%

props = [CrystalProperties.create(E=E, tK=tk) for tk in np.linspace(start, stop, steps)]
ref = CrystalProperties.create(E=E, tK=room)


# %%

# move by less than 1/10th the Darwin width
delta_theta = ref.darwin_width / 10


# %%
def f(T, delta):
    prop = CrystalProperties.create(E=E, tK=T)
    return delta - (ref.bragg_angle - prop.bragg_angle)


vlines = {}

# %%
for frac in [10, 100]:
    delta_theta = ref.darwin_width / frac

    lower = optimize.root_scalar(f, args=(delta_theta,), bracket=[start, stop])
    upper = optimize.root_scalar(f, args=(-delta_theta,), bracket=[start, stop])

    vlines[frac] = (lower.root, upper.root)

# %%
fig, ax = plt.subplots(layout="constrained")

ax.plot(
    [p.crystal.tK - room for p in props],
    [(p.bragg_angle - ref.bragg_angle) * 1000 * 1000 for p in props],
    label="Si 111",
    color="k",
)

trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
for (frac, (l, h)), style in zip(
    vlines.items(), [{"ls": ":"}, {"ls": "--"}], strict=True
):
    ax.axvline(l - room, color=".7", **style)
    ax.axvline(h - room, color=".7", **style)
    ax.text(
        h - room,
        0.05,
        f"{1e6 * ref.darwin_width / frac:.2f} μdeg (Darwin/{frac})",
        transform=trans,
        rotation=90,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=10,
        alpha=0.7,
    )
    ax.text(
        l - room,
        0.95,
        f"[{h - room:.2f}, {l - room:.2f}] K",
        transform=trans,
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=10,
        alpha=0.7,
    )
ax.set_xlim(-6, 6)
ax.set_xlabel(rf"$\Delta T$ from {room} (K)")
ax.set_ylabel(r"$\Delta \theta_{bragg}$ (μdeg)")
plt.show()
