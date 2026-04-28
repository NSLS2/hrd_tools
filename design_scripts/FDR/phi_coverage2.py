# %% [markdown]
# # Effective ɸ for detectors
#
# This file generates a plot showing the effective ɸ (of the DS cone)
# covered by various commercial detectors as a function of 2θ
#
# This is intended as a figure in HRD FDR
#
# in contrast to phi_coverage.py this uses the full correction rather than
# estimating via simple geometry.  The effect is small with a perfectly
# aligned crystal, but grows if chi is non-zero.  Not bothering to explore
# that here as the dominating factor is detector size.

# %%
import matplotlib.pyplot as plt
import numpy as np
from multihead.config import AnalyzerConfig
from multihead.corrections import arm_from_z

import _fdr_params
from hrd_tools.detector_stats import detectors

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)


# %%
def effoctive_solid_angle(tth, config, detector_width):
    # radius of DS ring at detector
    (arm_tth), (phi) = arm_from_z(
        np.array([detector_width / 2]).reshape(1, -1),
        np.array(tth).reshape(-1, 1),
        config,
    )
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
# Pull blessed (R, Rd, theta_i, theta_d) from beamline_params.yaml.
cfg = _fdr_params.analyzer_multihead(energy_keV=_args.energy_keV)

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

_save(fig, "phi_coverage_full.png")

# %% [markdown]
# This shows that the maximum ɸ coverage is effectively independent of the
# location of the crystal.

# %%
# effect of moving crystal position on max phi
# Figure 3
theta = np.linspace(1, 90)

# Total source-to-detector distance from blessed values.
_a = _fdr_params.load()["analyzer"]
_total = _a["R"] + _a["Rd"]

cfgs = [
    _fdr_params.analyzer_multihead(
        energy_keV=_args.energy_keV,
        crystal_roll=1 / 1000,
        R=middle,
        Rd=_total - middle,
    )
    for middle in [100, 250, 500, 900, 1000]
]

max_phi = np.array([effoctive_solid_angle(theta, cfg, 10).squeeze() for cfg in cfgs])

fig, (ax, ax2) = plt.subplots(
    2, 1, layout="constrained", figsize=(4, 4.5), dpi=100, sharex=True
)

ax.plot(
    theta,
    np.ptp(max_phi, axis=0),
    label="10 mm detector (ptp)",
    color="C0",
    lw=2,
)

for i, (cfg, middle) in enumerate(zip(cfgs, [100, 250, 500, 900, 1000])):
    ax2.plot(
        theta,
        max_phi[i],
        label=f"{middle} mm",
        alpha=0.6,
        lw=1,
    )

ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax2.set_xlabel(r"$2\theta$ (deg)")
ax.set_ylabel(r"$\Delta\phi_{max}$ (deg)")
ax2.set_ylabel(r"$\pm\phi_{max}$ (deg)")

ax.set_xlim(0, 90)

fig.suptitle(
    "Peak-to-peak variation in $\\pm\\phi_{max}$ for crystal positions 100-1000 mm"
)

_save(fig, "phi_coverage_crystal_position.png")
_fdr_params.maybe_show(_args)
