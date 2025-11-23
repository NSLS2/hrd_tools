import matplotlib.pyplot as plt
import numpy as np
import xrt.backends.raycing.materials as rmats
import matplotlib as mpl

E = 30_000
crystal = rmats.CrystalSi(t=1)

# in mdeg
darwin_mdeg = 1000 * np.rad2deg(crystal.get_Darwin_width(E))


def delta_L(arm_tth, delta_theta):
    arm_tth = np.deg2rad(arm_tth)
    delta_theta = np.deg2rad(delta_theta)
    return np.sin(arm_tth) * np.tan(np.pi / 2 - arm_tth + delta_theta) - np.cos(arm_tth)


cmap = mpl.colormaps["viridis"]
fig, ax = plt.subplots(layout="constrained")
delta_theta = np.linspace(-0.001, 0.001, 512)
R = 0.910 * 1e6

for arm_tth in [1, 5, 15, 25, 45, 90][::-1]:
    ax.plot(
        1000 * delta_theta,
        R * delta_L(arm_tth, delta_theta),
        label=rf"$2\Theta={arm_tth:.1f}\degree$",
        color=cmap(arm_tth / 90),
    )

ax.set_xlabel(r"$\delta \theta_i$ (mdeg)")
ax.set_ylabel(r"$\delta L$ (μm)")
ax.axhspan(-5, 5, color="r", alpha=0.1)
ax.axhspan(-40, 40, color="orange", alpha=0.1)
ax.axvspan(-darwin_mdeg / 2, darwin_mdeg / 2, color="k", alpha=0.1)
ax.set_title(f"At {E / 1000}kEv and {R / 1e6}m")
ax.set_ylim(-50, 50)
ax.legend()
plt.show()
