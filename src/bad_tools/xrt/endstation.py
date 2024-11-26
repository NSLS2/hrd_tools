from typing import Self

from dataclasses import dataclass
import functools

import numpy as np
import pandas as pd

import xrt.backends.raycing as raycing
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.materials as rmats

from bad_tools.config import AnalyzerConfig, SimConfig, SourceConfig, DetectorConfig
from bad_tools.xrt.sources import XrdSource
from bad_tools.xrt.stops import RectangularBeamstop


@dataclass
class Endstation:
    bl: raycing.BeamLine
    analyzer: AnalyzerConfig
    source: SourceConfig
    detector: DetectorConfig
    sim: SimConfig

    @classmethod
    def from_configs(
        cls,
        config: AnalyzerConfig,
        source_config: SourceConfig,
        detector_config: DetectorConfig,
        sim_config: SimConfig,
    ) -> Self:
        crystalSi01 = rmats.CrystalSi(t=config.thickness)

        theta_b = _bragg(crystalSi01, source_config.E_incident)
        arm_tth = np.deg2rad(15)
        beamLine = raycing.BeamLine()

        reference_pattern = pd.read_csv(
            source_config.pattern_path,
            skiprows=3,
            names=["theta", "I1", "I0"],
            sep=" ",
            skipinitialspace=True,
            index_col=False,
        )
        delta_phi = source_config.delta_phi
        beamLine.geometricSource01 = XrdSource(
            bl=beamLine,
            center=[0, 0, 0],
            dx=source_config.dx,
            dz=source_config.dz,
            dy=source_config.dy,
            distxprime=r"annulus",
            # this must be a sequence, but values are ignored
            dxprime=[0, 0],
            distzprime=r"flat",
            dzprime=[np.pi / 2 - delta_phi, np.pi / 2 + delta_phi],
            distE="normal",
            energies=[
                source_config.E_incident,
                source_config.E_incident * source_config.E_hwhm,
            ],
            pattern=reference_pattern,
            nrays=sim_config.nrays,
        )
        # TODO switch to plates
        beamLine.screen_main = rscreens.Screen(
            bl=beamLine, center=[0, 150, r"auto"], name="main"
        )

        for j in range(config.N):
            cry_tth = arm_tth + j * config.cry_offset
            # accept xrt coordinates
            cry_y = config.R * np.cos(cry_tth)
            cry_z = config.R * np.sin(cry_tth)

            # TODO These are all wrong but are fixed up by set_crystals
            pitch = -cry_tth + theta_b

            baffle_tth = arm_tth + (j + 0.5) * config.cry_offset
            baffle_pitch = np.pi / 4 - (2 * theta_b - baffle_tth)

            baffle_y = (
                config.R * np.cos(baffle_tth) + config.Rd / 2 * np.cos(baffle_pitch),
            )
            baffle_z = config.R * np.sin(baffle_tth) - config.Rd / 2 * np.sin(
                baffle_pitch
            )

            theta_pp = np.pi / 4 - (2 * theta_b - cry_tth)

            setattr(
                beamLine,
                f"oe{j:02d}",
                roes.OE(
                    name=f"cry{j:02d}",
                    bl=beamLine,
                    center=[0, cry_y, cry_z],
                    pitch=pitch,
                    positionRoll=np.pi,
                    material=crystalSi01,
                    limPhysX=[-config.cry_width / 2, config.cry_width / 2],
                    limPhysY=[-config.cry_depth / 2, config.cry_depth / 2],
                ),
            )
            setattr(
                beamLine,
                f"baffle{j:02d}",
                RectangularBeamstop(
                    name=f"baffle{j:02d}",
                    bl=beamLine,
                    opening=[
                        -config.cry_width / 2,
                        config.cry_width / 2,
                        -0.7 * config.Rd / 2,
                        0.7 * config.Rd / 2,
                    ],
                    center=[0, baffle_y, baffle_z],
                    z=(
                        0,
                        np.sin(baffle_pitch - np.pi / 2),
                        np.cos(baffle_pitch - np.pi / 2),
                    ),
                ),
            )
            screen_angle = theta_pp
            setattr(
                beamLine,
                f"screen{j:02d}",
                rscreens.Screen(
                    bl=beamLine,
                    center=[
                        0,
                        cry_y + config.Rd * np.cos(theta_pp),
                        cry_z - config.Rd * np.sin(theta_pp),
                    ],
                    x=(1, 0, 0),
                    z=(0, np.sin(screen_angle), np.cos(screen_angle)),
                ),
            )

        return cls(beamLine, config, source_config, detector_config, sim_config)

    @property
    def crystals(self):
        return [oe for oe in self.bl.oes if oe.name.startswith("cry")]

    @property
    def baffles(self):
        return [oe for oe in self.bl.slits if oe.name.startswith("baffle")]

    def set_arm(self, arm_tth: float):
        # def set_crystals(arm_tth: float, crystals, baffles, screens, config: AnalyzerConfig):
        config = self.analyzer
        crystals = self.crystals
        baffles = self.baffles
        screens = self.bl.screens[1:]

        theta_b = _bragg(crystals[0].material, self.source.E_incident)

        offset = config.cry_offset
        for j, (cry, baffle, screen) in enumerate(
            zip(crystals, baffles, screens, strict=True)
        ):
            cry_tth = arm_tth + j * offset
            # accept xrt coordinates
            cry_y = config.R * np.cos(cry_tth)
            cry_z = config.R * np.sin(cry_tth)
            pitch = -cry_tth + theta_b

            cry.center = [0, cry_y, cry_z]
            cry.pitch = pitch

            theta_pp = theta_b + pitch

            baffle_tth = arm_tth + (j - 0.5) * offset
            baffle_pitch = 2 * theta_b - baffle_tth

            baffle_y = (
                config.R * np.cos(baffle_tth) + config.Rd / 2 * np.cos(baffle_pitch),
            )
            baffle_z = config.R * np.sin(baffle_tth) - config.Rd / 2 * np.sin(
                baffle_pitch
            )

            baffle.center = [0, baffle_y, baffle_z]
            baffle.z = (
                0,
                np.sin(baffle_pitch - np.pi / 2),
                np.cos(baffle_pitch - np.pi / 2),
            )

            screen.center = [
                0,
                cry_y + config.Rd * np.cos(theta_pp),
                cry_z - config.Rd * np.sin(theta_pp),
            ]

            screen_angle = theta_pp

            screen.z = (0, np.sin(screen_angle), np.cos(screen_angle))


@functools.lru_cache(50)
def _bragg(mat, energy):
    return mat.get_Bragg_angle(energy)
