import numpy as np
import xrt.backends.raycing.materials as rmats
from multihead.config import AnalyzerConfig
from multihead.corrections import tth_from_z

from hrd_tools.detector_stats import detectors

crystal = rmats.CrystalSi(t=1)


def footprint(d: float, tth: float, E: float):
    bragg = crystal.get_Bragg_angle(E * 1000)
    tth = np.deg2rad(tth)

    return np.sin(np.pi - tth) * d / np.sin(bragg)


print("Z footprint")

E = 40
for max_z in [3, 5]:
    print(
        f"At {E}kEv with a {max_z:}mm sample requires a footprint of {footprint(max_z, 90, E):.2f} mm"
    )


for max_z in [3, 5]:
    print(
        f"{E}kEv\t{max_z:} mm \t {footprint(max_z, 90, E):.2f} mm"
    )

E = 30
for max_z in [3, 5]:
    print(
        f"At {E}kEv with a {max_z:}mm sample requires a footprint of {footprint(max_z, 90, E):.2f} mm"
    )


for max_z in [3, 5]:
    print(
        f"{E}kEv\t{max_z:} mm \t {footprint(max_z, 90, E):.2f} mm"
    )


cfg = AnalyzerConfig(
    910,  # R: sample to crystal distance (mm)
    120,  # Rd: crystal to detector distance (mm)
    np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),  # theta_i: incident angle
    2 * np.rad2deg(np.arcsin(0.8 / (2 * 3.1355))),  # theta_d: diffraction angle
    detector_roll=0,
)

# in mm
det_size = dict(
    sorted(
        {
            k: det.pixel_pitch * det.sensor_shape[1] / 1000
            for k, det in detectors.items()
        }.items(),
        key=lambda x: x[1],
    )
)

arm_angles = np.array([45])
z = np.array(list(det_size.values())) / 2
corrected_tths, corrected_phis = tth_from_z(
    z.reshape(1, -1), arm_angles.reshape(-1, 1), cfg
)

on_cry_size = (
    2 * cfg.R * np.sin(np.deg2rad(corrected_tths)) * np.sin(np.deg2rad(corrected_phis))
)

print("X footprint")
for (det_name, det_width), cry_z in zip(
    det_size.items(), on_cry_size.squeeze(), strict=True
):
    print(
        f"{det_name: <8} ({det_width:.2f} mm wide) "
        f"require {cry_z:.2f} mm strain free crystal width"
    )

for (det_name, det_width), cry_z in zip(
    det_size.items(), on_cry_size.squeeze(), strict=True
):
    print(
        f"{det_name: <8}\t{det_width:.2f} mm\t"
        f"{cry_z:.2f} mm"
    )
