"""Required crystal footprint (longitudinal & transverse) per energy/detector."""

import numpy as np
import xrt.backends.raycing.materials as rmats
from multihead.corrections import tth_from_z

import _fdr_params
from hrd_tools.detector_stats import detectors

_args = _fdr_params.parse_args(__doc__)
_blessed = _fdr_params.complete_config()

crystal = rmats.CrystalSi(t=_fdr_params.crystal_reference()["thickness_mm"])


def footprint(d: float, tth: float, E: float):
    bragg = crystal.get_Bragg_angle(E * 1000)
    tth = np.deg2rad(tth)

    return np.sin(np.pi - tth) * d / np.sin(bragg)


print("Z footprint")

E = 40                                         # keV
for max_z in [3, 5]:                            # mm
    print(
        f"At {E}keV with a {max_z:}mm sample requires a footprint of {footprint(max_z, 90, E):.2f} mm"
    )

for max_z in [3, 5]:
    print(f"{E}keV\t{max_z:} mm \t {footprint(max_z, 90, E):.2f} mm")

E = 30                                         # keV
for max_z in [3, 5]:
    print(
        f"At {E}keV with a {max_z:}mm sample requires a footprint of {footprint(max_z, 90, E):.2f} mm"
    )

for max_z in [3, 5]:
    print(f"{E}keV\t{max_z:} mm \t {footprint(max_z, 90, E):.2f} mm")


cfg = _fdr_params.analyzer_multihead()

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

arm_angles = np.array([45])                    # deg
z = np.array(list(det_size.values())) / 2      # mm
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
    print(f"{det_name: <8}\t{det_width:.2f} mm\t{cry_z:.2f} mm")
