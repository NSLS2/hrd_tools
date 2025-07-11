# %%
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import xrt.backends.raycing.materials as rmats


# %%
def angle(n1, v1):
    return np.arccos(
        np.vecdot(n1, v1)
        / (np.linalg.vector_norm(n1) * np.linalg.vector_norm(v1, axis=1))
    )


def to_norm(θ, ɸ):
    return np.array(
        [
            np.sin(θ) * np.cos(ɸ),
            np.sin(θ) * np.sin(ɸ),
            np.cos(θ) * np.ones_like(ɸ),
        ]
    )


# %%
E = 30_000
# TODO generate with other material, hkl
crystal = rmats.CrystalSi(t=1, hkl=(1, 1, 1))
bragg = crystal.get_Bragg_angle(E)
darwin = crystal.get_Darwin_width(E)
phi = np.deg2rad(np.linspace(0, 1, 1024))

# %%
#
# fig, ax = plt.subplots(layout="constrained")
#
# for arm in np.deg2rad([15, 45, 60, 90]):
#     n1 = to_norm(arm, 0).T
#     v = to_norm(arm, phi).T
#
#     ax.plot(
#         np.rad2deg(phi),
#         np.rad2deg(angle(n1[:, np.newaxis,:], v[np.newaxis, :, :)),
#         label=f"$2\\theta = {np.rad2deg(arm):.1f}\\degree$",
#     )
# ax.set_title(f"at {E / 1_000}kEv")
# ax.annotate(
#     f"Si 111 Darwin width\n({np.rad2deg(darwin):.2g}$\\degree$)",
#     (0, np.rad2deg(darwin)),
#     xytext=(5, 5),
#     textcoords="offset pixels",
# )
# ax.axhline(np.rad2deg(darwin))
# ax.set_xlim(0, 1)
# # ax.set_ylim(0, np.rad2deg(darwin) * 3)
# ax.set_xlabel(r"$\phi$ (deg)")
# ax.set_ylabel(r"$\theta_{bragg} - 2\theta_{eff}$ (deg)")
# ax.legend()
#
# %%

phi = np.deg2rad(np.linspace(0, 1, 1024))
out = []
arm_th = np.linspace(-1, 6, 5120)
for arm in np.deg2rad(arm_th):
    n1 = to_norm(arm - bragg, 0)
    v = to_norm(arm, phi).T
    out.append(np.rad2deg(0 * bragg - (angle(n1, v))))

angle_deviation = np.asarray(out)


# 1m sample -> crystal
d = 1
# 320*75um for medipix4
detector_width = 320 * 75 * 10e-6

# 256*55um for medipix3
# detector_width = 256 * 55 * 10e-6

# 1028*75um for eiger2
# detector_width = 1028* 75 * 10e-6


def effoctive_solid_angle(theta, d, detector_width):
    # radius of DS ring at detector
    R = d * np.abs(np.sin(theta))
    return 2 * np.arctan2(detector_width / 2, R)


sa = effoctive_solid_angle(np.deg2rad(arm_th), d, detector_width)

fig, ax = plt.subplots(layout="constrained")
ax.plot(arm_th, sa, label=r"max $\phi$ on detector")
ax.plot(
    arm_th,
    np.rad2deg(phi[np.argmin(np.abs(angle_deviation) < darwin, axis=1)]),
    label="max $\phi$ from Darwin",
)
ax.legend()
ax.set_title(f"sample-to-detector {d}m, detector_width {detector_width * 1000:.1f}mm")

# %%
plt.figure()
plt.imshow(
    # np.abs(angle_deviation) < darwin,
    bragg - angle_deviation,
    aspect="auto",
    extent=(
        np.rad2deg(np.min(phi)),
        np.rad2deg(np.max(phi)),
        np.min(arm_th),
        np.max(arm_th),
    ),
    origin="lower",
)
plt.colorbar()

# %%


fig, ax = plt.subplots(layout="constrained")

ring = 25
arm = np.linspace(ring - 0.01, ring + 0.01, 512)
phi = np.linspace(-0.10, 0.10, 256)
n1 = to_norm(np.deg2rad(arm), 0).T
v = to_norm(np.deg2rad(ring), phi).T

angles = np.arccos(np.vecdot(v[np.newaxis, :, :], n1[:, np.newaxis, :]))
# angles = np.vecdot(v[np.newaxis, :, :], n1[:, np.newaxis, :])

plt.imshow(
    angles,
    aspect="auto",
    extent=[np.min(phi), np.max(phi), np.min(ring - arm), np.max(ring - arm)],
)

# %%

# Demonstration of the full ring passing note that the arm angle to cover the
# full range is smaller at small angles so at a fixed phi range look "flatter"
# for a fixed movement of the arm despite having higher curvature


def plot_ring_pass(ring, ax):
    arm = np.linspace(0 * ring - 0.02, ring + 0.01, 512 * 2)
    phi = np.linspace(-90, 90, 256 * 2)
    n1 = to_norm(np.deg2rad(arm), 0).T
    v = to_norm(np.deg2rad(ring), np.deg2rad(phi)).T

    ax.set_title(f"{ring=}")
    # angles = np.arccos(np.vecdot(v[np.newaxis, :, :], n1[:, np.newaxis, :]))
    return ax.imshow(
        np.abs(
            np.atan(n1[:, 0] / n1[:, 2])[:, np.newaxis]
            - np.atan(v[:, 0] / v[:, 2])[np.newaxis, :]
        )
        < darwin * 100,
        origin="lower",
        extent=[np.min(phi), np.max(phi), ring - np.min(arm), ring - np.max(arm)],
        aspect="auto",
    )


fig, ax_lst = plt.subplots(3, layout="constrained")
for ring, ax in zip([1, 25, 45], ax_lst):
    plot_ring_pass(ring, ax)
# %%

# This shows what we would expect to see on a fixed sized detector at several
# angles.  Countering the affect above, the restriction of the accssible phi at
# higher angles makes them look flatavailable phqter


def effoctive_solid_angle(theta, d, detector_width):
    # radius of DS ring at detector
    R = d * np.abs(np.sin(theta))
    return np.arctan2(detector_width / 2, R)


def plot_ring_pass(ring, ax, d_width, d_R):
    arm = np.linspace(ring - 0.1, ring + 0.01, 512 * 2)
    dphi = effoctive_solid_angle(np.deg2rad(np.mean(arm)), d_R, d_width)
    phi = np.rad2deg(np.linspace(-dphi, dphi, 256 * 2))
    n1 = to_norm(np.deg2rad(arm), 0).T
    v = to_norm(np.deg2rad(ring), np.deg2rad(phi)).T

    ax.set_title(f"{ring=}")
    angles = np.arccos(np.vecdot(v[np.newaxis, :, :], n1[:, np.newaxis, :]))
    return ax.imshow(
        np.abs(
            np.atan(n1[:, 0] / n1[:, 2])[:, np.newaxis]
            - np.atan(v[:, 0] / v[:, 2])[np.newaxis, :]
        )
        < darwin,
        origin="lower",
        extent=[np.min(phi), np.max(phi), ring - np.min(arm), ring - np.max(arm)],
        aspect="auto",
    )
    # ax.plot(v[:, 0])


dets = {
    # 320*75um for medipix4
    "medipix4": 320 * 75 * 1e-6,
    # 256*55um for timepix4
    "timepix4": 512 * 55 * 1e-6,
    # 256*55um for medipix3, timepix3
    "medipix3": 256 * 55 * 1e-6,
    # 1028*75um for eiger2
    "eiger2": 1028 * 75 * 1e-6,
}

fig, ax_lst = plt.subplots(3, layout="constrained", sharex=True)
for ring, ax in zip([75, 45, 15], ax_lst):
    plot_ring_pass(ring, ax, dets["medipix4"], 1)

# %%


def plotz(ax_lst, vec):
    for ax, data in zip(ax_lst, vec.T):
        ax.plot(data)


fig, ax_lst = plt.subplots(3)
plotz(ax_lst, n1)
