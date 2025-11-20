# %%
# # Concentricity Analysis
#
# This notebook is to aid specifying how concentric the detector mounting must be with the center
# of rotation of the diffractometer.

# %%

import numpy as np
import matplotlib.pyplot as plt
# %%
# At any given point on the analyzer crystal there is a line along with
# scattered rays will perfectly match the Bragg condition of the crystal
# and be passed through that point.  Because the crystals are wide transverse
# to the beam, there is a plane of scattering locations that will each pass
# through a line on on the crystal.
#
# For scattering locations displaced from this plane by small distances the
# rays may still be passed if the displacement is small enough that the
# deviation of the ray is less than the Darwin width of the crystal will still be
# passed.
#
# The size of the region that will be accepted is larger the farther the scattering
# location is from the crystal, thus the gauge volume is a wedge shape.

# %%
import xrt.backends.raycing.materials as rmats
E = 25_000
crystal = rmats.CrystalSi(t=1)
bragg = crystal.get_Bragg_angle(E)
# in radians
darwin = crystal.get_Darwin_width(E)
# %%

def gauge_width(R: float, energy: float) -> float:
    """Calculate the width of the gauge volume at a given distance from the crystal.

    The gauge volume is determined by the Darwin width of the crystal and increases
    with distance from the crystal surface, forming a wedge shape.

    Parameters
    ----------
    R : float
        Distance from the crystal surface in the same units as desired for the output
    energy : float
        X-ray energy in eV

    Returns
    -------
    float
        Width of the gauge volume at distance R from the crystal surface
    """
    crystal = rmats.CrystalSi(t=1)
    darwin = crystal.get_Darwin_width(energy)
    return R * np.tan(darwin)

# %%
# Plot gauge width vs energy
energies = np.linspace(10_000, 40_000, 1000)
widths = [gauge_width(1000000, E) for E in energies]

fig = plt.figure(figsize=(10, 6), layout='constrained')
ax = fig.add_subplot(111)
ax.plot(energies/1000, widths)  # Convert to keV for plotting
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Gauge Width (um)')
ax.set_title('Inherent Gauge Width at R=1m vs Energy')
ax.grid(True)
fig.show()

# Plot gauge width vs distance at 30kEV

R = np.linspace(0, 10_000_000, 512)
widths = gauge_width(R, 30_000)

fig = plt.figure(figsize=(10, 6), layout='constrained')
ax = fig.add_subplot(111)
ax.plot(R/1000, widths)  # Convert to keV for plotting
ax.set_xlabel('R (mm)')
ax.set_ylabel('Gauge Width (um)')
ax.set_title('Inherent Gauge Width at 30kEv vs distance')
ax.grid(True)
fig.show()
# %%
# This displacement is perpendicular to the plane of ideal scattering, thus as the crystal is rotated
# around the sample, there is a volume of sample that is always visible to the detector.
# Thus, for a perfectly aligned crystal with an infinitely thin detector a scan will sample a sphere
# of the sample that is measured at every position and a shell of locations that are sampled at only
# a small range of crystal positions.  If the center of the circle the detector are on is not concentric
# with the rotation axis of the diffractometer the volume sampled at all positions will be hollowed out
# to a shell until there is no consistently sampled volume.  Thus we want the two circles to be concentric
# to within 2 um (1/3 the gauge width at 40keV an 1m)

# %%
# The above analysis only considers
