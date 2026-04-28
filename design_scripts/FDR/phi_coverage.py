# %% [markdown]
# # Effective ɸ for detectors
#
# This file generates a plot showing the effective ɸ (of the DS cone)
# covered by various commercial detectors as a function of θ
#
# This is intended as a figure in HRD FDR
#
# in contrast to phi_coverage2.py this uses simple geometry to estimate
# effective ɸ coverage rather than the full correction.

# %%
import matplotlib.pyplot as plt
import numpy as np

import _fdr_params
from hrd_tools.detector_stats import detectors

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)


# %%
def effoctive_solid_angle(theta, d, detector_width):
    # radius of DS ring at detector
    R = d * np.abs(np.sin(theta))
    return np.arctan2(detector_width / 2, R)


# %%
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
# sample -> detector distance from blessed analyzer geometry (R + Rd, mm -> m)
_a = _fdr_params.load()["analyzer"]
d = (_a["R"] + _a["Rd"]) / 1000.0
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
ax.set_title(f"$\\pm\\phi_{{max}}$ at {d:.2f} m")

_save(fig, "phi_coverage_simple.png")
_fdr_params.maybe_show(_args)
