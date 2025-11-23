# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
def to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)

    # theta = arccos(z / (r)
    theta = np.arccos(z / r)
    phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))

    return r, theta, phi


def line_path(start, stop, npt=1024):
    return tuple(
        np.linspace(_start, _stop, npt)
        for _start, _stop in zip(start, stop, strict=True)
    )


# %%
start, stop = (50, 0, 1200), (200, 1000, -200)
x, y, z = line_path(start, stop)
r, theta, phi = to_spherical(x, y, z)

d = np.cumsum(
    np.hstack([0, np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)])
)

fig, (ax_r, ax_th, ax_phi) = plt.subplots(
    3, layout="constrained", sharex=True, figsize=(3, 6)
)

fig.suptitle(f"{start} to {stop}")
for data, ax, label in zip(
    (r, np.rad2deg(theta), np.rad2deg(phi)),
    (ax_r, ax_th, ax_phi),
    ("R (mm)", r"$2\theta$ (deg)", r"$\phi$ (deg)"),
    strict=True,
):
    ax.plot(d, data)
    ax.set_ylabel(label)

ax_phi.set_xlabel("along detector (mm)")
ax_phi.set_xlim(xmin=0)
plt.show()
