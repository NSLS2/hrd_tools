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
    min_tth = 7.3
    max_tth = 8.7
    return CompleteConfig(
        **{
            "source": SourceConfig(
                E_incident=29_400,
                pattern_path="bob",
                dx=4,
                dz=0.01,
                dy=0,
                delta_phi=16,
                E_hwhm=1.4e-4,
                h_div=np.rad2deg(0),
                v_div=np.rad2deg(0),
                max_tth=max_tth,
                min_tth=min_tth,
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
            "scan": SimScanConfig(start=min_tth, stop=max_tth, delta=1e-4),
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
        cycler("source.pattern_path", ["/nsls2/users/tcaswell/11bmb_7871_Y1.xye"])
        * cycler("analyzer.acceptance_angle", np.array([.1, .2, 1, 2]) * 0.05651551)
    )
    for j, config in enumerate(configs):
        with open(f"config_{j}.toml", "wb") as fout:
            tomli_w.dump(asdict(config), fout)
