# %% [markdown]
# # Sensitivity to χ missalignment on 2ϴ to 2θ correction


# %%
import dataclasses

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from multihead.config import AnalyzerConfig
from multihead.corrections import tth_from_z
from scipy import optimize

from hrd_tools.xrt import CrystalProperties

# %%
mpl.rcParams["savefig.dpi"] = 300

# %% [markdown]
#
# The error gets large with both the displacment across the detector
# and the detector arm angle

# %%

props = CrystalProperties.create(E=40)

# Configure analyzer with realistic parameters
cfg = AnalyzerConfig(
    910,  # R: sample to crystal distance (mm)
    120,  # Rd: crystal to detector distance (mm)
    props.bragg_angle,
    2 * props.bragg_angle,
    detector_roll=0,
)

z = 15


fig, (ax1, ax2) = plt.subplots(1, 2, layout="constrained", sharey=True)


def one_z(z, ax):
    arm_angle = np.linspace(2, 88)
    baseline, _ = tth_from_z(z, arm_angle, cfg)
    for chi in [0.0001, 0.001, 0.01]:
        corrected_tths, _ = tth_from_z(
            z, arm_angle, dataclasses.replace(cfg, crystal_roll=chi)
        )
        (ln,) = ax.plot(
            arm_angle, (baseline - corrected_tths) * 1000, label=rf"$\chi$={chi}"
        )
        corrected_tths, _ = tth_from_z(
            -z, arm_angle, dataclasses.replace(cfg, crystal_roll=chi)
        )
        ax.plot(arm_angle, (baseline - corrected_tths) * 1000, color=ln.get_color())

    ax.axhline(1e-1, color=".5", ls="--")
    ax.axhline(-1e-1, color=".5", ls="--")

    ax.legend(loc="best")
    ax.set_title(rf"$z_d$={z}mm")
    ax.set_xlabel(r"arm $2\Theta$ (deg)")


def one_angle(arm_angle, ax):
    z = np.linspace(-20, 20, 256)
    baseline, _ = tth_from_z(z, arm_angle, cfg)
    for chi in [0.0001, 0.001, 0.01]:
        corrected_tths, _ = tth_from_z(
            z, arm_angle, dataclasses.replace(cfg, crystal_roll=chi)
        )
        ax.plot(z, (baseline - corrected_tths) * 1000, label=rf"$\chi$={chi}")

    ax.axhline(1e-1, color=".5", ls="--")
    ax.axhline(-1e-1, color=".5", ls="--")

    ax.legend()
    ax.set_title(rf"$2\Theta$={arm_angle}deg")
    ax.set_xlabel(r"arm $2\Theta$ (deg)")


ax1.set_ylabel(r"scatter $\Delta 2\theta$ (mdeg)")

one_z(z, ax2)
one_angle(45, ax1)


plt.show()

# %%

zd = 15
arm_angle = 88
delta_tth = 1e-5

baseline_tth, _ = tth_from_z(zd, arm_angle, cfg)


def f(chi, delta):
    corrected_tth, _ = tth_from_z(
        zd, arm_angle, dataclasses.replace(cfg, crystal_roll=chi)
    )

    return delta - (-corrected_tth + baseline_tth)


chi_limit = optimize.root_scalar(f, args=(delta_tth,), bracket=[0, 0.01])

print(
    f"at z_d={zd}mm and 2ϴ={arm_angle}deg the maximum χ "
    f"to stay under Δ2θ ≤ {delta_tth * 1000:.2g}mdeg "
    f"is {1000 * chi_limit.root:.2g}mdeg"
)
