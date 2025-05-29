import functools
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import xrt.backends.raycing as raycing
import xrt.backends.raycing.screens as rscreens

from hrd_tools.config import SimConfig, SourceConfig
from hrd_tools.xrt.sources import XrdSource


@dataclass(frozen=True)
class StripDetectorConfig:
    # distance from source
    radius: float
    # strip width (the pixelated direction) in mm
    strip_width: float
    # strip height (the "big" wai) in mm
    strip_height: float
    # location of center, offset to miss-align
    center: tuple[float, float, float] = (0, 0, 0)


@dataclass
class Endstation:
    bl: raycing.BeamLine
    source: SourceConfig
    detector: StripDetectorConfig
    sim: SimConfig

    @classmethod
    def from_configs(
        cls,
        source: SourceConfig,
        detector: StripDetectorConfig,
        sim: SimConfig,
    ) -> Self:
        beamLine = raycing.BeamLine()

        reference_pattern = pd.read_csv(
            source.pattern_path,
            skiprows=3,
            names=["theta", "I1", "I0"],
            sep=" ",
            skipinitialspace=True,
            index_col=False,
        )
        delta_phi = np.deg2rad(source.delta_phi)
        beamLine.geometricSource01 = XrdSource(
            bl=beamLine,
            center=[0, 0, 0],
            dx=source.dx,
            dz=source.dz,
            dy=source.dy,
            distxprime=r"annulus",
            dxprime=[source.min_tth, source.max_tth],
            distzprime=r"flat",
            dzprime=[np.pi / 2 - delta_phi, np.pi / 2 + delta_phi],
            distE="normal",
            energies=[
                source.E_incident,
                source.E_incident * source.E_hwhm,
            ],
            pattern=reference_pattern,
            nrays=sim.nrays,
            horizontal_divergence=source.h_div,
            vertical_divergence=source.v_div,
        )
        # TODO switch to plates
        beamLine.screen_main = rscreens.Screen(
            bl=beamLine, center=[0, 150, r"auto"], name="main"
        )

        beamLine.strip_sphere = rscreens.HemisphericScreen(
            beamLine,
            "strip_sphere",
            detector.center,
            R=detector.radius,
            # x=(0, -np.sin(thetaOffset), np.cos(thetaOffset)),
            # z=(0, np.cos(thetaOffset), np.sin(thetaOffset)),
        )

        return cls(beamLine, source, detector, sim)

    def run_process(self):
        # "raw" beam
        beamLine = self.bl
        geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()
        screen01beamLocal01 = beamLine.screen_main.expose(
            beam=geometricSource01beamGlobal01
        )
        screen02beamLocal01 = beamLine.strip_sphere.expose(
            beam=geometricSource01beamGlobal01
        )

        return {"strip_screen": screen02beamLocal01}

    def get_frame(self):
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

        out = {}

        for k, lb in screen_beams.items():
            # print(lb.x, lb.y, lb.z, lb.state)
            inner = {}
            for kt, good in zip(
                ("good",),
                (((lb.state == 1) | (lb.state == 2)),),
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

    def get_free_ray_image(self):
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
                ("bad",),
                (free_rays,),
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
