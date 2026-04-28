# %%
# # Concentricity Analysis
#
# This notebook is to aid specifying how concentric the detector mounting must be with the center
# of rotation of the diffractometer.

"""Gauge-volume width vs distance and energy → concentricity spec."""

# %%

import matplotlib.pyplot as plt
import numpy as np

import _fdr_params
from hrd_tools.xrt import CrystalProperties

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)
_blessed = _fdr_params.complete_config()
_e_keV = _args.energy_keV if _args.energy_keV is not None else _blessed.source.E_incident / 1000.0
_R_mm = _blessed.analyzer.R                     # mm

print(f"{plt.get_backend()=}")

# %%
# At any given point on the analyzer crystal there is a line along which
# scattered rays will perfectly match the Bragg condition of the crystal
# and be passed through that point.  Because the crystals are wide transverse
# to the beam, there is a plane of scattering locations that will each pass
# through a line on the crystal.
#
# For scattering locations displaced from this plane by small distances the
# rays may still be passed if the displacement is small enough that the
# deviation of the ray is less than the Darwin width of the crystal.
#
# The size of the region that will be accepted is larger the farther the scattering
# location is from the crystal, thus the gauge volume is a wedge shape.
# %%
crystal_props = CrystalProperties.create(E=_e_keV)


# %%


def gauge_width(R: float, energy: float) -> float:
    """Calculate the width of the gauge volume at a given distance from the crystal.

    The gauge volume is determined by the Darwin width of the crystal and increases
    with distance from the crystal surface, forming a wedge shape.

    Parameters
    ----------
    R : float
        Distance from the crystal surface (any length unit; output is in same unit)
    energy : float
        X-ray energy in keV

    Returns
    -------
    float
        Width of the gauge volume at distance R from the crystal surface
    """
    props = CrystalProperties.create(E=energy)
    return R * np.tan(np.radians(props.darwin_width))


# %%
# Plot gauge width vs energy at R = blessed analyzer.R
energies = np.linspace(10, 40, 1000)            # keV
_R_um = _R_mm * 1000.0                          # µm
widths = [gauge_width(_R_um, E) for E in energies]

fig = plt.figure(figsize=(10, 6), layout="constrained")
ax = fig.add_subplot(111)
ax.plot(energies, widths)
ax.set_xlabel("Energy (keV)")
ax.set_ylabel("Gauge Width (um)")
ax.set_title(f"Inherent Gauge Width at R={_R_mm:.0f} mm vs Energy")
ax.grid(True)
_save(fig, "concentricity_vs_energy.png")

# Plot gauge width vs distance at canonical energy
R = np.linspace(0, 10_000_000, 512)              # µm
widths = gauge_width(R, _e_keV)

fig = plt.figure(figsize=(10, 6), layout="constrained")
ax = fig.add_subplot(111)
ax.plot(R / 1000, widths)
ax.set_xlabel("R (mm)")
ax.set_ylabel("Gauge Width (um)")
ax.set_title(f"Inherent Gauge Width at {_e_keV:.0f} keV vs distance")
ax.grid(True)
_save(fig, "concentricity_vs_distance.png")

# %%
# This displacement is perpendicular to the plane of ideal scattering, thus as the crystal is rotated
# around the sample, there is a volume of sample that is always visible to the detector.
# Thus, for a perfectly aligned crystal with an infinitely thin detector a scan will sample a sphere
# of the sample that is measured at every position and a shell of locations that are sampled at only
# a small range of crystal positions.  If the center of the circle the detector are on is not concentric
# with the rotation axis of the diffractometer the volume sampled at all positions will be hollowed out
# to a shell until there is no consistently sampled volume.  Thus we want the two circles to be concentric
# to within ~2 µm (1/3 the gauge width at 40 keV at 1 m).

_fdr_params.maybe_show(_args)
print("bye")
