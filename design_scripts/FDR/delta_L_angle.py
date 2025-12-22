# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xrt.backends.raycing.materials as rmats

# %%

mpl.rcParams['savefig.dpi'] = 300

# %%
E = 40_000
crystal = rmats.CrystalSi(t=1)

# in mdeg
darwin_mdeg = 1000 * np.rad2deg(crystal.get_Darwin_width(E))


# %%
def delta_L(R, arm_tth, delta_theta):
    arm_tth = np.deg2rad(arm_tth)
    delta_theta = np.deg2rad(delta_theta)
    return R * (
        np.sin(arm_tth) * np.tan(np.pi / 2 - arm_tth + delta_theta) - np.cos(arm_tth)
    )


# %%
cmap = mpl.colormaps["viridis"]
fig, (ax, ax2) = plt.subplots(1, 2, layout="constrained", sharey=True)
fig.suptitle(f"At {E / 1000}kEv and {R / 1e6}m")
delta_theta = np.linspace(-0.001, 0.001, 512)
R = 0.9 * 1e6

for arm_tth in [1.5, 5, 15, 25, 45, 90][::-1]:
    ax.plot(
        1e6 * delta_theta,
        delta_L(R, arm_tth, delta_theta),
        label=rf"$2\Theta={arm_tth:.1f}\degree$",
        color=cmap(arm_tth / 90),
    )

ax.set_xlabel(r"$\delta \theta_i$ (μdeg)")
ax.set_ylabel(r"$\delta L$ (μm)")
ax.axhspan(-5, 5, color="k", alpha=0.1)
ax.axhspan(-40, 40, color="k", alpha=0.1)
# ax.axvspan(-darwin_mdeg / 2, darwin_mdeg / 2, color="k", alpha=0.1)
ax.axvline(-1000 * darwin_mdeg / 2, color="k", alpha=0.5, zorder=1, ls="--")
ax.axvline(1000 * darwin_mdeg / 2, color="k", alpha=0.5, zorder=1, ls="--")

ax.set_ylim(-50, 50)
ax.set_xlim(-2 * 1000 * darwin_mdeg, 2 * 1000 * darwin_mdeg)
ax.legend()

arm_tths = np.linspace(1.5, 90, 512)
cmap = mpl.colormaps["plasma"]
ax2.set_xlabel(r"$2\Theta$ (deg)")
for delta_theta in [darwin_mdeg / (1000 * _) for _ in (2, 5, 10, 100)]:
    (ln,) = ax2.plot(
        arm_tths,
        delta_L(R, arm_tths, delta_theta),
        label=rf"$\delta \theta_i$ = {1e6 * delta_theta:.2f} (μdeg)",
        color=cmap(2 * delta_theta / (darwin_mdeg / 1000)),
    )
    ax2.plot(arm_tths, delta_L(R, arm_tths, -delta_theta), color=ln.get_color())
ax2.axhspan(-5, 5, color="k", alpha=0.1)
ax2.axhspan(-40, 40, color="k", alpha=0.1)
ax2.legend()
ax2.set_xlim(1.5, 90)
plt.show()

# %%
