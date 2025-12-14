# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from hrd_tools.xrt import CrystalProperties

# %%
room = 295.0  # K
start, stop = room - 15, room + 15  # K

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


lower = optimize.root_scalar(f, args=(delta_theta,), bracket=[start, stop])
upper = optimize.root_scalar(f, args=(-delta_theta,), bracket=[start, stop])
# %%
fig, ax = plt.subplots(layout="constrained")

ax.set_title(
    f"<{delta_theta * 1e6:.2f}μdeg Bragg shift requires [{upper.root - room:.2f}, {lower.root - room:.2f}]K stability"
)
ax.plot(
    [p.crystal.tK - room for p in props],
    [(p.bragg_angle - ref.bragg_angle) * 1000 * 1000 for p in props],
    label="Si 111",
)

ax.axhspan(
    -delta_theta * 1e6,
    delta_theta * 1e6,
    color="r",
    alpha=0.5,
    label=rf"$\pm {delta_theta * 1e6:.2f}$ (μdeg)",
)
ax.legend()
ax.set_xlabel(rf"$\Delta T$ from {room} (K)")
ax.set_ylabel(r"$\Delta \theta_{bragg}$ (μdeg)")
plt.show()
