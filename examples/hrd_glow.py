# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:45:07 2026

@author: roman
"""
import numpy as np
import xrt.backends.raycing.run as rrun
from pathlib import Path

from hrd_tools.config import AnalyzerConfig, SimConfig, SourceConfig
from hrd_tools.xrt.endstation_canonical import build_beamline, run_process


def main():
    analyzer = AnalyzerConfig(
        R=300,
        Rd=115,
        cry_offset=np.deg2rad(2.5),
        cry_width=102,
        cry_depth=54,
        N=3,
        incident_angle=0.0,  # overwritten from Bragg angle in Endstation wrapper only
        thickness=1,
    )

    source = SourceConfig(
        E_incident=29_400,
        pattern_path=Path(r"C:\GitHub\hrd_tools\examples\11bmb_7871_Y1.xye"),
        dx=1,
        dz=0.1,
        dy=0,
        delta_phi=5,  # degrees
        E_hwhm=1.4e-4,
        min_tth=14,
        max_tth=24,
    )

#    sim = SimConfig(nrays=3_000_000)
    sim = SimConfig(nrays=10_000)

    hrd = build_beamline(analyzer, source, sim)
    rrun.run_process = run_process
    scene_settings = {'apertureBladeWidth': 102/4, 'apertureDefaultSpan': 115,
                      'rayFlag': {1}}
    hrd.glow(sceneSettings=scene_settings)


if __name__ == "__main__":
    main()
