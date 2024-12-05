from dataclasses import asdict

import numpy as np
import tomli_w

from bad_tools.config import (
    AnalyzerConfig,
    DetectorConfig,
    SimConfig,
    SimScanConfig,
    SourceConfig,
)


def get_defaults():
    return {
        "source": asdict(
            SourceConfig(
                E_incident=29_400,
                pattern_path="bob",
                dx=1,
                dz=0.1,
                dy=0,
                delta_phi=np.pi / 8,
                E_hwhm=1.4e-4,
            )
        ),
        "sim": asdict(SimConfig(nrays=100_000)),
        "detector": asdict(DetectorConfig(pitch=0.055, transverse_size=512, height=1)),
        "analyzer": asdict(
            AnalyzerConfig(
                R=300,
                Rd=115,
                cry_offset=np.deg2rad(2.5),
                cry_width=102,
                cry_depth=54,
                N=3,
                acceptance_angle=0.05651551,
                thickness=1,
            )
        ),
        "scan": asdict(SimScanConfig(start=5, stop=15, delta=1e-4)),
    }


if __name__ == "__main__":
    with open("test.toml", "wb") as fout:
        tomli_w.dump(
            get_defaults(),
            fout,
        )
