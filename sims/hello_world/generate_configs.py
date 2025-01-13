from collections import defaultdict
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import tomli_w
from cycler import Cycler, cycler

from bad_tools.config import (
    AnalyzerConfig,
    CompleteConfig,
    DetectorConfig,
    SimConfig,
    SimScanConfig,
    SourceConfig,
)


def get_defaults():
    return CompleteConfig(
        **{
            "source": SourceConfig(
                E_incident=29_400,
                pattern_path="bob",
                dx=1,
                dz=0.1,
                dy=0,
                delta_phi=16,
                E_hwhm=1.4e-4,
                h_div=np.rad2deg(0),
                v_div=np.rad2deg(0.0002),
                max_tth=9,
                min_tth=7,
            ),
            "sim": SimConfig(nrays=1_000_000),
            "detector": DetectorConfig(pitch=0.055, transverse_size=512, height=1),
            "analyzer": AnalyzerConfig(
                R=300,
                Rd=115,
                cry_offset=np.deg2rad(2.5),
                cry_width=102,
                cry_depth=54,
                N=1,
                acceptance_angle=0.05651551,
                thickness=1,
            ),
            "scan": SimScanConfig(start=7, stop=9, delta=1e-4),
        }
    )


def convert_cycler(cycle: Cycler) -> list[CompleteConfig]:
    defaults = get_defaults()
    out = []
    for entry in cycle:
        nested: dict[str, dict[str, Any]] = defaultdict(dict)
        for k, v in entry.items():
            outer, _, inner = k.partition(".")
            nested[outer][inner] = v

        for k, v in nested.items():
            out.append(replace(defaults, **{k: replace(getattr(defaults, k), **v)}))
    return out


if __name__ == "__main__":
    configs = convert_cycler(
        cycler("source.pattern_path", ["/home/tcaswell/Downloads/11bmb_7871_Y1.xye"])
        * cycler("source.v_div", np.rad2deg(np.array([0, 0.02, 0.2, 2]) / 1000))
    )
    for j, config in enumerate(configs):
        with open(f"config2_{j}.toml", "wb") as fout:
            tomli_w.dump(asdict(config), fout)
