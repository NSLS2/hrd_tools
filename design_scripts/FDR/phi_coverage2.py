# %% [maarkdown]
# # Effective ɸ for detectors
#
# This file generates a plot showing the effective ɸ (of the DS cone)
# covered by various commercial detectors as a function of 2θ
#
# This is intended as a figure in HRD FDR
#
# in contrast to phi_coverage.py this uses the full correction rather than estimating
# via simple geometry.  The effect is small with a perfectly aligned crystal, but grows
# if chi is non-zero.  Not bothering to explore that here as the dominating factor is detector
# size.

# %%

import matplotlib.pyplot as plt
import numpy as np
from multihead.config import AnalyzerConfig
from multihead.corrections import arm_from_z

from hrd_tools.detector_stats import detectors


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


# %% [markdown]
# ## Plot effective ɸ for various detectors
#


# Calculate detector widths from detector stats
dets = {
    name: det.sensor_shape[1] * det.pixel_pitch * 1e-6  # convert µm to meters
    for name, det in detectors.items()
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


# %% [markdown]
# This shows that the maximum ɸ coverage is effectively independent of the location
# of the crystal.
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
