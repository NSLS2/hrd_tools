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
from multihead.config import AnalyzerConfig
from multihead.corrections import arm_from_z


# %%
def effoctive_solid_angle(tth, config, detector_width):
    # radius of DS ring at detector
    (arm_tth), (phi) = arm_from_z(
        np.array([detector_width / 2]).reshape(1, -1),
        np.array(tth).reshape(-1, 1),
        config,
    )
    print(arm_tth, phi)
    return phi


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

# 1.03m sample -> detector
cfg = AnalyzerConfig(
    910,
    120,
    np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
    2 * np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
    detector_roll=0,
)

theta = np.linspace(1, 90)

fig, ax = plt.subplots(layout="constrained", figsize=(4, 2.75), dpi=100)
for det, width in sorted(dets.items(), key=lambda x: x[1]):
    sa = effoctive_solid_angle(theta, cfg, width * 1e3).squeeze()

    ax.plot(
        theta,
        sa,
        label=f"{det} ({width * 100:.2f} cm)",
        **style.get(det, {}),
    )

ax.legend()
ax.set_xlabel(r"$2\theta$ (deg)")
ax.set_ylabel(r"$\pm\phi_{max}$ (deg)")
ax.set_ylim(0, 4)
ax.set_xlim(0, 90)

ax.set_title(f"$\\pm\\phi_{{max}}$ at {cfg.R + cfg.Rd:.2f} mm")

plt.show()


# %%


theta = np.linspace(1, 90)

cfgs = [
    AnalyzerConfig(
        middle,
        1030 - middle,
        np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
        2 * np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),
        # crystal_roll=-0.01,
    )
    for middle in [250, 500, 900]
]

fig, ax = plt.subplots(layout="constrained", figsize=(4, 2.75), dpi=100)
for cfg in cfgs:
    sa = effoctive_solid_angle(theta, cfg, 10).squeeze()

    ax.plot(
        theta,
        sa,
        label=f"{cfg.R=}",
    )

ax.legend()
ax.set_xlabel(r"$2\theta$ (deg)")
ax.set_ylabel(r"$\pm\phi_{max}$ (deg)")
ax.set_ylim(0, 4)
ax.set_xlim(0, 90)

ax.set_title(f"$\\pm\\phi_{{max}}$ at {cfg.R + cfg.Rd:.2f} mm")

plt.show()
