from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import multianalyzer.opencl
import numpy as np
from matplotlib.figure import Figure
from multianalyzer import Result

from bad_tools.config import (
    AnalyzerCalibration,
    AnalyzerConfig,
    CompleteConfig,
    DetectorConfig,
    SimScanConfig,
)


def load_all_config(
    path: Path, *, ext="h5", prefix=""
) -> dict[Path, tuple[CompleteConfig, AnalyzerCalibration] | None]:
    configs = {}
    for _ in path.glob(f"**/{prefix}*{ext}"):
        config = load_config(_)

        configs[_] = config
    return configs


def load_config(fname, *, tlg="sim"):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        configs = {}

        for fld in fields(CompleteConfig):
            try:
                config_grp = g[f"{fld.name}_config"]
            except KeyError:
                print(f"missing {fld.name}")
            else:
                configs[fld.name] = fld.type(**config_grp.attrs)
        if "scan" not in configs:
            tth = g["tth"][:]
            configs["scan"] = SimScanConfig(
                start=np.rad2deg(tth[0]),
                stop=np.rad2deg(tth[-1]),
                delta=np.rad2deg(np.mean(np.diff(tth))),
            )
        complete_config = CompleteConfig(**configs)

    calibration = AnalyzerCalibration(
        detector_centers=(
            [complete_config.detector.transverse_size / 2] * complete_config.analyzer.N
        ),
        psi=[
            np.rad2deg(complete_config.analyzer.cry_offset) * j
            for j in range(complete_config.analyzer.N)
        ],
    )
    return complete_config, calibration


def find_varied_config(configs):
    out = defaultdict(set)
    for c in configs:
        cd = asdict(c)
        for k, sc in cd.items():
            for f, v in sc.items():
                out[(k, f)].add(v)
    return [k for k, v in out.items() if len(v) > 1]


def load_data(fname, *, tlg="sim", scale=1):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        block = g["block"][:]
        block *= scale
        return np.rad2deg(g["tth"][:]), block.astype("uint32")


def reduce_raw(
    block,
    tths,
    tth_min: float,
    tth_max: float,
    dtth: float,
    analyzer: AnalyzerConfig,
    detector: DetectorConfig,
    calibration: AnalyzerCalibration,
    phi_max: float = 90,
    mode="opencl",
):
    if mode == "cython":
        cls = multianalyzer.MultiAnalyzer
    elif mode == "opencl":
        cls = multianalyzer.opencl.OclMultiAnalyzer
        cls.NUM_CRYSTAL = np.int32(analyzer.N)
    else:
        raise ValueError
    mma = cls(
        # sample to crystal
        L=analyzer.R,
        # crystal to detector
        L2=analyzer.Rd,
        # pixel size
        pixel=detector.pitch,
        # pixel column direct beam is on
        # TODO pull from calibration structure
        center=np.array(calibration.detector_centers),
        # acceptance angle of crystals
        tha=analyzer.acceptance_angle,
        # global offset of MCA vs arm position
        # TODO pull from calibration structure
        thd=0.0,
        # offsets of each crystal from origin of MCA
        # TODO pull from calibration structure
        psi=calibration.psi,
        # mis-orientation of the analyzer along x (°)
        # TODO pull from calibration structure
        rollx=[0.0] * analyzer.N,
        # TODO pull from calibration structure
        # mis-orientation of the analyzer along y (°)
        rolly=[0.0] * analyzer.N,
    )
    return mma.integrate(
        # data all stacked as one
        roicollection=block,
        # arm positions
        arm=tths,
        # IO, currently constant
        mon=np.ones_like(tths),
        # range and resolution for binning
        tth_min=np.float64(tth_min),
        tth_max=np.float64(tth_max),
        dtth=np.float64(dtth),
        # shape of "ROIs"
        num_row=detector.transverse_size,
        num_col=1,
        # how to understand the shape of roicollection
        columnorder=1,
        phi_max=phi_max,
    )


def sum_mca(mca):
    point_data = mca.sum(axis=1)
    return (point_data - point_data.min()) / np.ptp(point_data)


def plot_raw(tth, mca, title):
    row, col = mca.shape
    if (row,) != tth.shape:
        raise RuntimeError

    fig, ax_d = plt.subplot_mosaic(
        [["parasite", "mca"]], width_ratios=(1, 5), layout="constrained", sharey=True
    )

    fig.suptitle(title)
    ax_d["mca"].imshow(
        mca, extent=(0, col, tth[0], tth[-1]), origin="lower", aspect="auto"
    )
    ax_d["mca"].set_xlabel("channel index")
    point_data = sum_mca(mca)
    max_indx = np.argmax(point_data)
    max_tth = tth[max_indx]
    ax_d["parasite"].axhline(max_tth, color="k", alpha=0.5, zorder=0)
    ax_d["parasite"].annotate(
        f"{max_tth:.2f}°",
        xy=(0, max_tth),
        xytext=(0, -5),
        textcoords="offset points",
        ha="left",
        va="top",
    )
    ax_d["parasite"].plot(point_data, tth)
    ax_d["parasite"].set_xlabel("column sum (arb)")
    ax_d["parasite"].set_ylabel("arm angle (deg)")
    return ax_d


def reduce_file(
    fname: Path,
    config: CompleteConfig,
    calib: AnalyzerCalibration,
    mode: str = "opencl",
):
    dtth = config.scan.delta
    print(f"{dtth=}")
    tth, channels = load_data(fname)
    # for j in range(config.analyzer.N):
    #     plot_raw(tth, channels[:, j, :, :].squeeze(), f"crystal {j}")
    ret = reduce_raw(
        block=channels,
        tths=tth.astype("float64"),
        tth_min=np.float64(config.scan.start),
        tth_max=config.scan.stop
        + config.analyzer.N * np.rad2deg(config.analyzer.cry_offset),
        dtth=dtth,
        analyzer=config.analyzer,
        detector=config.detector,
        calibration=calib,
        phi_max=np.float64(90),
        mode=mode,
    )
    # fig, ax = plt.subplots()
    # for j in range(config.analyzer.N):
    #     break
    #     ax.plot(ret.tth, 0.1 * j + ret.signal[j, :, 0] / ret.norm[j], label=j)
    # ax.legend()
    return ret


def reduce_and_plot(
    configs: dict[Path, tuple[CompleteConfig, AnalyzerCalibration]],
) -> tuple[dict[Path, Result], Figure]:
    res = {k: reduce_file(k, *v) for k, v in configs.items()}
    label_keys = find_varied_config([a for a, _ in configs.values()])
    fig, ax = plt.subplots()
    for k in list(res):
        ret = res[k]
        config = configs[k][0]
        for j in range(config.analyzer.N):
            label = " ".join(
                f"{sec}.{parm}={getattr(getattr(config, sec), parm):.02g}"
                for sec, parm in label_keys
            )
            normed = ret.signal[j, :, 0] / ret.norm[j]
            mask = np.isfinite(normed)
            ax.plot(ret.tth[mask], 0.1 * j + normed[mask], label=label)
    ax.legend()
    return res, fig
