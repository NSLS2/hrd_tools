import functools
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import pandas as pd
import xrt.backends.raycing as raycing
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.screens as rscreens
#import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.apertures as rapts

from ..config import AnalyzerConfig, DetectorConfig, SimConfig, SourceConfig
from .sources import XrdSource
from .stops import RectangularBeamstop

DEFAULT_ARM_TTH = np.deg2rad(15)


class AnalyzerBeamLine(raycing.BeamLine):
    """XRT beamline for the multi-analyzer HRD endstation."""

    def __init__(
        self,
        analyzer: AnalyzerConfig,
        source: SourceConfig,
        sim: SimConfig,
        *,
        arm_tth: float = DEFAULT_ARM_TTH,
    ) -> None:
        super().__init__()
        self.analyzer_config = analyzer
        self.source_config = source
        self.sim_config = sim
        self.crystal_material = rmats.CrystalSi(t=analyzer.thickness)
        self.theta_b = _bragg(self.crystal_material, source.E_incident)

        self.source = make_source(self, source, sim)
        self.screen_main = rscreens.Screen(
            bl=self, center=[0, 150, r"auto"], name="main"
        )

        self.crystals = []
        self.baffles = []
        self.detector_screens = []
        self._build_analyzers()
        self.set_arm(arm_tth)

        # Backwards-compatible aliases used by older notebooks and glow helpers.
        self.geometricSource01 = self.source
        for j, crystal in enumerate(self.crystals):
            setattr(self, f"oe{j:02d}", crystal)
        for j, baffle in enumerate(self.baffles):
            setattr(self, f"baffle{j:02d}", baffle)
        for j, screen in enumerate(self.detector_screens):
            setattr(self, f"screen{j:02d}", screen)

    def _build_analyzers(self) -> None:
        config = self.analyzer_config
        for j in range(config.N):
            crystal = roes.OE(
                name=f"cry{j:02d}",
                bl=self,
                center=[0, 0, 0],
                pitch=0,
#                pitch='auto',
                positionRoll=np.pi,
                material=None, #self.crystal_material,
                limPhysX=[-config.cry_width / 2, config.cry_width / 2],
                limPhysY=[-config.cry_depth / 2, config.cry_depth / 2],
                extraRoll=np.deg2rad(config.roll),
            )
            baffle = RectangularBeamstop(
                name=f"baffle{j:02d}",
                bl=self,
#                opening=[
#                    -config.cry_width / 2,
#                    config.cry_width / 2,
#                    -0.7 * config.Rd / 2,
#                    0.7 * config.Rd / 2,
#                ],
                opening=[-0.01, 0.01, -0.01, 0.01],
                center=[0, 0, 0],
                z=(0, 0, 1),
            )
            screen = rscreens.Screen(
                name=f"screen{j:02d}",
                bl=self,
                center=[0, 0, 0],
                x=(1, 0, 0),
                z=(0, 0, 1),
            )
            self.crystals.append(crystal)
            self.baffles.append(baffle)
            self.detector_screens.append(screen)

    def set_arm(self, arm_tth: float) -> None:
        config = self.analyzer_config
        theta_b = self.theta_b

        for j, (crystal, baffle, screen) in enumerate(
            zip(self.crystals, self.baffles, self.detector_screens, strict=True)
        ):
            cry_tth = arm_tth + j * config.cry_offset
            cry_y = config.R * np.cos(cry_tth)
            cry_z = config.R * np.sin(cry_tth)
            pitch = -cry_tth + theta_b

            crystal.center = [0, cry_y, cry_z]
            crystal.pitch = pitch

            theta_pp = theta_b + pitch

            baffle_tth = arm_tth + (j - 0.5) * config.cry_offset
            baffle_pitch = 2 * theta_b - baffle_tth

            baffle.center = [
                0,
                config.R * np.cos(baffle_tth) + config.Rd / 2 * np.cos(baffle_pitch),
                config.R * np.sin(baffle_tth) - config.Rd / 2 * np.sin(baffle_pitch),
            ]
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
            screen.z = (0, np.sin(theta_pp), np.cos(theta_pp))


@dataclass
class Endstation:
    bl: AnalyzerBeamLine
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
        crystal = rmats.CrystalSi(t=analyzer.thickness)
        analyzer = replace(
            analyzer,
            incident_angle=np.rad2deg(_bragg(crystal, source.E_incident)),
        )
        beamline = build_beamline(analyzer, source, sim)
        return cls(beamline, analyzer, source, detector, sim)

    @property
    def crystals(self):
        return self.bl.crystals

    @property
    def baffles(self):
        return self.bl.baffles

    def set_arm(self, arm_tth: float) -> None:
        self.bl.set_arm(arm_tth)

    def run_process(self):
        return _screen_outputs(run_process(self.bl))

    def get_frames(self):
        detector_config = self.detector
        screen_beams = self.run_process()

        shape, limits = _detector_histogram_geometry(detector_config)
        out = {}
        _, yedges, xedges = np.histogram2d([], [], bins=shape, range=limits)
        for key, beam in screen_beams.items():
            good = (beam.state == 1) | (beam.state == 2)
            hist2d, yedges, xedges = _histogram_screen(beam, good, shape, limits)
            out[key] = {"good": hist2d}

        return out, yedges, xedges

    def get_free_ray_image(self):
        detector_config = self.detector
        screen_beams = self.run_process()

        shape, limits = _detector_histogram_geometry(detector_config)
        states = np.vstack([v.state for v in screen_beams.values()])
        (free_ray_indx,) = np.where((states == 3).sum(axis=0) == len(screen_beams))
        free_rays = np.zeros(states.shape[1], dtype=bool)
        free_rays[free_ray_indx] = True

        out = {}
        for key, beam in screen_beams.items():
            hist2d, yedges, xedges = _histogram_screen(beam, free_rays, shape, limits)
            out[key] = {"bad": hist2d}

        return out, yedges, xedges


def build_beamline(
    analyzer: AnalyzerConfig,
    source: SourceConfig,
    sim: SimConfig,
    *,
    arm_tth: float = DEFAULT_ARM_TTH,
) -> AnalyzerBeamLine:
    return AnalyzerBeamLine(analyzer, source, sim, arm_tth=arm_tth)


def run_process(beamLine):
    source_beam = beamLine.source.shine()
    source_screen = beamLine.screen_main.expose(beam=source_beam)

    out = {
        "source": source_beam,
        "source_screen": source_screen,
    }

    for crystal, baffle, screen in zip(
        beamLine.crystals,
        beamLine.baffles,
        beamLine.detector_screens,
        strict=True,
    ):
        reflected_global, reflected_local = crystal.reflect(beam=source_beam)
        out[f"{crystal.name}_local"] = reflected_local
        out[f"{crystal.name}_global"] = reflected_global
#        out[f"{baffle.name}_local"] = baffle.propagate(beam=reflected_global)
        out[screen.name] = screen.expose(beam=reflected_global)

    return out


def make_source(
    beamline: raycing.BeamLine,
    source: SourceConfig,
    sim: SimConfig,
) -> XrdSource:
    delta_phi = np.deg2rad(source.delta_phi)
    return XrdSource(
        bl=beamline,
        name="source",
        center=(
            source.source_offset_x,
            source.source_offset_y,
            source.source_offset_z,
        ),
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
        pattern_path=source.pattern_path,
        nrays=sim.nrays,
        horizontal_divergence=source.h_div,
        vertical_divergence=source.v_div,
    )


def load_reference_pattern(source: SourceConfig) -> pd.DataFrame:
    return pd.read_csv(
        source.pattern_path,
        skiprows=3,
        names=["theta", "I1", "I0"],
        sep=" ",
        skipinitialspace=True,
        index_col=False,
    )


def _screen_outputs(beams):
    return {key: value for key, value in beams.items() if key.startswith("screen")}


def _detector_histogram_geometry(detector_config: DetectorConfig):
    shape = (
        int(detector_config.height // detector_config.pitch),
        detector_config.transverse_size,
    )
    limits = list(
        (detector_config.pitch * np.array([[-0.5, 0.5]]).T * np.array([shape])).T
    )
    return shape, limits


def _histogram_screen(beam, good, shape, limits):
    flux = beam.Jss[good] + beam.Jpp[good]
    return np.histogram2d(
        beam.z[good], beam.x[good], bins=shape, range=limits, weights=flux
    )


@functools.lru_cache(50)
def _bragg(mat, energy):
    return mat.get_Bragg_angle(energy) + mat.get_dtheta(energy)
