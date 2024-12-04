from dataclasses import dataclass
from pathlib import Path

# TODO look into this game
# from typing import NewType
# rad = NewType("rad", float)
# mm = NewType("mm", float)


@dataclass(frozen=True)
class AnalyzerConfig:
    # sample to (central) crystal
    R: float
    # crystal to detector distance
    Rd: float
    # angular offset between crystals in deg
    cry_offset: float
    # crystal width (transverse to beam) in mm
    cry_width: float
    # crystal depth (direction of beam) in mm
    cry_depth: float
    # number of crystals
    N: int
    # acceptance angle of crystals
    acceptance_angle: float
    # thickness of crystals in mm
    thickness: float


@dataclass(frozen=True)
class DetectorConfig:
    # pixel pitch in mm
    pitch: float
    # pixel width in transverse direction
    transverse_size: int
    # Size of active area in direction of beam in mm
    height: float


@dataclass(frozen=True)
class SimConfig:
    # number of rays in the simulation
    nrays: int


@dataclass(frozen=True)
class SourceConfig:
    # Energy
    E_incident: float
    # location of source pattern
    # TODO maybe make actually pattern?
    pattern_path: Path
    # width of source in mm
    dx: float
    # height of source in mm
    dz: float
    # depth of source in mm
    dy: float
    # phi range around vertical to simulate
    delta_phi: float
    E_hwhm: float
