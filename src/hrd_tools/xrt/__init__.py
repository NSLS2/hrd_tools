from dataclasses import dataclass
from typing import Self

import numpy as np
import xrt.backends.raycing.materials as rmats

from .sources import XrdSource

__all__ = ["XrdSource", "CrystalProperties"]


@dataclass
class CrystalProperties:
    """Container for computed crystal properties at a specific energy.

    Attributes
    ----------
    crystal : rmats.CrystalSi
        The underlying XRT CrystalSi object
    bragg_angle : float
        The Bragg angle in degrees
    darwin_width : float
        The Darwin width in degrees
    energy_kev : float
        The X-ray energy in keV
    """

    crystal: rmats.CrystalSi
    bragg_angle: float
    darwin_width: float
    energy_kev: float

    @classmethod
    def create(
        cls,
        E: float,
        t: float = 1.0,
        hkl: tuple[int, int, int] = (1, 1, 1),
        tK: float = 293.15,
    ) -> Self:
        """Create a CrystalProperties instance from energy and crystal configuration.

        Parameters
        ----------
        E : float
            X-ray energy in keV
        t : float, optional
            Crystal thickness in mm, by default 1.0
        hkl : tuple[int, int, int], optional
            Miller indices, by default (1, 1, 1)
        tK : float, optional
            Crystal temperature in Kelvin, by default 293.15 (room temperature)

        Returns
        -------
        CrystalProperties
            Instance containing the crystal object and precomputed properties
        """
        energy_ev = E * 1000
        crystal = rmats.CrystalSi(hkl=hkl, t=t, tK=tK)
        bragg_angle = crystal.get_Bragg_angle(energy_ev)
        darwin_width = crystal.get_Darwin_width(energy_ev)

        return cls(
            crystal=crystal,
            bragg_angle=np.degrees(bragg_angle),
            darwin_width=np.degrees(darwin_width),
            energy_kev=E,
        )

    def with_energy(self, energy_kev: float) -> Self:
        """Generate a new instance with a different energy.

        Parameters
        ----------
        energy_kev : float
            The new X-ray energy in keV

        Returns
        -------
        CrystalProperties
            New instance with updated energy and recomputed properties
        """
        return self.create(
            E=energy_kev,
            t=self.crystal.t,
            hkl=self.crystal.hkl,
            tK=self.crystal.tK,
        )
