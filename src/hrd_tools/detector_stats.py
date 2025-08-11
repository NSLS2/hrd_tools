from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


class DetectorMode(Enum):
    COUNTING = auto()
    PHOTON_STAMPING = auto()


@dataclass
class Detector:
    name: str
    pixel_pitch: float  # in µm
    sensor_shape: Tuple[int, int]  # (width, height) in pixels
    mode: DetectorMode


detectors = {
    "medipix4": Detector(
        name="Medipix4",
        pixel_pitch=75.0,
        sensor_shape=(320, 320),
        mode=DetectorMode.COUNTING,
    ),
    "timepix4": Detector(
        name="Timepix4",
        pixel_pitch=55.0,
        sensor_shape=(512, 448),
        mode=DetectorMode.PHOTON_STAMPING,
    ),
    "medipix3": Detector(
        name="Medipix3",
        pixel_pitch=55.0,
        sensor_shape=(256, 256),
        mode=DetectorMode.COUNTING,
    ),
    "eiger2": Detector(
        name="Eiger2",
        pixel_pitch=75.0,
        sensor_shape=(1028, 512),
        mode=DetectorMode.COUNTING,
    ),
}


