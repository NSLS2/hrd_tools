from typing import Self

from dataclasses import dataclass
import functools

import numpy as np
import pandas as pd

import xrt.backends.raycing as raycing
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.sources_beams as rsources_beams

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
        analyzer: AnalyzerConfig,
        source: SourceConfig,
        detector: DetectorConfig,
        sim: SimConfig,
    ) -> Self:
        crystalSi01 = rmats.CrystalSi(t=analyzer.thickness)

        theta_b = _bragg(crystalSi01, source.E_incident)
        arm_tth = np.deg2rad(15)
        beamLine = raycing.BeamLine()

        reference_pattern = pd.read_csv(
            source.pattern_path,
            skiprows=3,
            names=["theta", "I1", "I0"],
            sep=" ",
            skipinitialspace=True,
            index_col=False,
        )
        delta_phi = source.delta_phi
        beamLine.geometricSource01 = XrdSource(
            bl=beamLine,
            center=[0, 0, 0],
            dx=source.dx,
            dz=source.dz,
            dy=source.dy,
            distxprime=r"annulus",
            # this must be a sequence, but values are ignored
            dxprime=[0, 0],
            distzprime=r"flat",
            dzprime=[np.pi / 2 - delta_phi, np.pi / 2 + delta_phi],
            distE="normal",
            energies=[
                source.E_incident,
                source.E_incident * source.E_hwhm,
            ],
            pattern=reference_pattern,
            nrays=sim.nrays,
        )
        # TODO switch to plates
        beamLine.screen_main = rscreens.Screen(
            bl=beamLine, center=[0, 150, r"auto"], name="main"
        )

        for j in range(analyzer.N):
            cry_tth = arm_tth + j * analyzer.cry_offset
            # accept xrt coordinates
            cry_y = analyzer.R * np.cos(cry_tth)
            cry_z = analyzer.R * np.sin(cry_tth)

            # TODO These are all wrong but are fixed up by set_crystals
            pitch = -cry_tth + theta_b

            baffle_tth = arm_tth + (j + 0.5) * analyzer.cry_offset
            baffle_pitch = np.pi / 4 - (2 * theta_b - baffle_tth)

            baffle_y = (
                analyzer.R * np.cos(baffle_tth) + analyzer.Rd / 2 * np.cos(baffle_pitch),
            )
            baffle_z = analyzer.R * np.sin(baffle_tth) - analyzer.Rd / 2 * np.sin(
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
                    limPhysX=[-analyzer.cry_width / 2, analyzer.cry_width / 2],
                    limPhysY=[-analyzer.cry_depth / 2, analyzer.cry_depth / 2],
                ),
            )
            setattr(
                beamLine,
                f"baffle{j:02d}",
                RectangularBeamstop(
                    name=f"baffle{j:02d}",
                    bl=beamLine,
                    opening=[
                        -analyzer.cry_width / 2,
                        analyzer.cry_width / 2,
                        -0.7 * analyzer.Rd / 2,
                        0.7 * analyzer.Rd / 2,
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
                        cry_y + analyzer.Rd * np.cos(theta_pp),
                        cry_z - analyzer.Rd * np.sin(theta_pp),
                    ],
                    x=(1, 0, 0),
                    z=(0, np.sin(screen_angle), np.cos(screen_angle)),
                ),
            )

        return cls(beamLine, analyzer, source, detector, sim)

    @property
    def crystals(self):
        return [oe for oe in self.bl.oes if oe.name.startswith("cry")]

    @property
    def baffles(self):
        return [oe for oe in self.bl.slits if oe.name.startswith("baffle")]

    def set_arm(self, arm_tth: float):
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

    def run_process(self):
        # "raw" beam
        beamLine = self.bl
        geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()
        screen01beamLocal01 = beamLine.screen_main.expose(
            beam=geometricSource01beamGlobal01
        )

        outDict = {
            "source": geometricSource01beamGlobal01,
            "source_screen": screen01beamLocal01,
        }
        N = len([oe for oe in beamLine.oes if oe.name.startswith("cry")])

        for j in range(N):
            oeglobal, oelocal = getattr(beamLine, f"oe{j:02d}").reflect(
                beam=geometricSource01beamGlobal01
            )
            outDict[f"cry{j:02d}_local"] = oelocal
            outDict[f"cry{j:02d}_global"] = oeglobal

            outDict[f"baffle{j:0d}_local"] = getattr(
                beamLine, f"baffle{j:02d}"
            ).propagate(beam=oeglobal)

            outDict[f"screen{j:02d}"] = getattr(beamLine, f"screen{j:02d}").expose(
                beam=oeglobal
            )

        return {k: v for k, v in outDict.items() if k.startswith("screen")}

    def get_frames(self):
        detector_config = self.detector
        screen_beams = self.run_process()

        isScreen = True

        shape = (
            int(detector_config.height // detector_config.pitch),
            detector_config.transverse_size,
        )

        limits = list(
            (detector_config.pitch * np.array([[-0.5, 0.5]]).T * np.array([shape])).T
        )

        states = np.vstack([v.state for v in screen_beams.values()])
        (free_ray_indx,) = np.where((states == 3).sum(axis=0) == len(screen_beams))
        free_rays = np.zeros(states.shape[1], dtype=bool)
        free_rays[free_ray_indx] = True
        out = {}

        for k, lb in screen_beams.items():
            # print(lb.x, lb.y, lb.z, lb.state)
            inner = {}
            for kt, good in zip(
                ("good", "bad"),
                (((lb.state == 1) | (lb.state == 2)), free_rays),
                strict=True,
            ):
                if isScreen:
                    x, y = lb.x[good], lb.z[good]
                else:
                    x, y = lb.x[good], lb.y[good]

                flux = lb.Jss[good] + lb.Jpp[good]
                hist2d, yedges, xedges = np.histogram2d(
                    y, x, bins=shape, range=limits, weights=flux
                )
                inner[kt] = hist2d
            out[k] = inner

        return out, yedges, xedges


@functools.lru_cache(50)
def _bragg(mat, energy):
    return mat.get_Bragg_angle(energy)
