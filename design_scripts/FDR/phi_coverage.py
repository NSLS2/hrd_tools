# %% [maarkdown]
# # Effective ɸ for detectors
#
# This file generates a plot showing the effective ɸ (of the DS cone)
# covered by various commercial detectors as a function of θ
#
# This is intended as a figure in HRD FDR

# %%

import matplotlib.pyplot as plt
import numpy as np


# %%
def effoctive_solid_angle(theta, d, detector_width):
    # radius of DS ring at detector
    R = d * np.abs(np.sin(theta))
    return np.arctan2(detector_width / 2, R)


# %%

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

style = {
    "medipix4": {"lw": 3, "ls": "-"},
    "eiger2": {"lw": 1, "ls": ":", "color": ".7"},
    "medipix3": {"lw": 1, "ls": "--", "color": ".7"},
}

# %%

# 1m sample -> crystal
d = 1.03
theta = np.linspace(0, np.deg2rad(120))

fig, ax = plt.subplots(layout="constrained", figsize=(4, 2.75), dpi=100)
for det, width in sorted(dets.items(), key=lambda x: x[1]):
    sa = effoctive_solid_angle(theta, d, width)
    ax.plot(
        np.rad2deg(theta),
        np.rad2deg(sa),
        label=f"{det} ({width * 100:.2f} cm)",
        **style.get(det, {}),
    )
ax.legend()
ax.set_xlabel(r"$2\theta$ (deg)")
ax.set_ylabel(r"$\pm\phi_{max}$ (deg)")
ax.set_ylim(0, 4)
ax.set_xlim(0, 120)
# ax.set_yscale('log')
ax.set_title(f"$\\pm\\phi_{{max}}$ at {d:.2f} m")

plt.show()
