from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np

from .config import (
    AnalyzerCalibration,
    CompleteConfig,
    SimScanConfig,
)


def load_all_config(path: Path, *, ext="h5", prefix="") -> dict[Path, CompleteConfig]:
    configs = {}
    for _ in sorted(path.glob(f"**/{prefix}*{ext}")):
        config = load_config(_)

        configs[_] = config
    return configs


def load_config(fname, *, tlg="sim"):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        return load_config_from_group(g)


def load_config_from_group(grp):
    configs = {}

    for fld in fields(CompleteConfig):
        try:
            config_grp = grp[f"{fld.name}_config"]
        except KeyError:
            if fld.name != "scan":
                print(f"missing {fld.name}")
        else:
            configs[fld.name] = fld.type(**config_grp.attrs)
    if "scan" not in configs:
        tth = grp["tth"][:]
        configs["scan"] = SimScanConfig(
            start=np.rad2deg(tth[0]),
            stop=np.rad2deg(tth[-1]),
            delta=np.rad2deg(np.mean(np.diff(tth))),
        )
    return CompleteConfig(**configs)


def dflt_config(complete_config):
    return AnalyzerCalibration(
        detector_centers=(
            [complete_config.detector.transverse_size / 2] * complete_config.analyzer.N
        ),
        psi=[
            np.rad2deg(complete_config.analyzer.cry_offset) * j
            for j in range(complete_config.analyzer.N)
        ],
        roll=[complete_config.analyzer.roll] * complete_config.analyzer.N,
    )


def load_data(fname, *, tlg="sim", scale=1):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        block = g["block"][:]
        block *= scale
        return np.rad2deg(g["tth"][:]), block.astype("int32")
