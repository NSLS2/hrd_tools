from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import multianalyzer.opencl
import numpy as np
from matplotlib.figure import Figure
from multianalyzer import Result

from .config import AnalyzerCalibration, AnalyzerConfig, DetectorConfig
from .file_io import dflt_config, find_varied_config, load_config, load_data


def reduce_raw(
    block,
    tths,
    *,
    tth_min: float,
    tth_max: float,
    dtth: float,
    analyzer: AnalyzerConfig,
    detector: DetectorConfig,
    calibration: AnalyzerCalibration,
    phi_max: float = 90,
    mode="opencl",
    width=0,
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
    frames, _, num_col, rox_max = block.shape
    ret = mma.integrate(
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
        roi_max=rox_max,
        # how to understand the shape of roicollection
        columnorder=1,
        phi_max=phi_max,
        width=width,
    )
    return ret


def sum_mca(mca):
    point_data = mca.sum(axis=1)
    return (point_data - point_data.min()) / np.ptp(point_data)


def plot_raw(
    tth,
    mca,
    title,
    *,
    fig: matplotlib.figure.Figure | None = None,
    label_max: bool = False,
):
    row, col = mca.shape
    if (row,) != tth.shape:
        raise RuntimeError

    if fig is None:
        fig = plt.figure(layout="constrained")

    ax_d = fig.subplot_mosaic([["parasite", "mca"]], width_ratios=(1, 5), sharey=True)

    fig.suptitle(title)
    im = ax_d["mca"].imshow(
        mca, extent=(0, col, tth[0], tth[-1]), origin="lower", aspect="auto"
    )
    ax_d["mca"].set_xlabel("channel index")
    point_data = sum_mca(mca)
    if label_max:
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
    cb = fig.colorbar(im, ax=ax_d["mca"])
    cb.set_label("counts")
    return ax_d


def reduce_file(
    fname: Path,
    calib: AnalyzerCalibration | None = None,
    mode: str = "opencl",
    dtth_scale: float = 5.0,
    phi_max: float = 90,
):
    config = load_config(fname)
    if calib is None:
        calib = dflt_config(config)
    dtth = config.scan.delta * dtth_scale
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
        phi_max=np.float64(phi_max),
        mode=mode,
        width=0,
    )
    # fig, ax = plt.subplots()
    # for j in range(config.analyzer.N):
    #     break
    #     ax.plot(ret.tth, 0.1 * j + ret.signal[j, :, 0] / ret.norm[j], label=j)
    # ax.legend()
    return ret, config, calib


def reduce_and_plot(
    files: list[Path],
    # configs: dict[Path, tuple[CompleteConfig, AnalyzerCalibration]],
    **kwargs,
) -> tuple[dict[Path, Result], Figure]:
    reduced = {k: reduce_file(k, **kwargs) for k in files}

    label_keys = find_varied_config([v[1] for v in reduced.values()])
    print(label_keys)
    fig, ax = plt.subplots()
    for _, (ret, config, _) in reduced.items():
        for j in range(config.analyzer.N):
            label = " ".join(
                f"{sec}.{parm}={getattr(getattr(config, sec), parm):.02g}"
                for sec, parm in label_keys
            )
            normed = ret.signal[j, :, 0] / ret.norm[j]
            mask = np.isfinite(normed)
            ax.plot(ret.tth[mask], 0.1 * j + normed[mask], label=label)
    ax.legend()
    return reduced, fig


def normalize_result(res: Result, scale_to_max=True):
    signal = res.signal.sum(axis=2)
    out = np.zeros_like(signal, dtype=float)
    # first dimension is number of crystals
    for j in range(signal.shape[0]):
        normed = signal[j] / res.norm[j]
        if scale_to_max:
            normed /= np.nanmax(normed)
        out[j] = normed
    return out


def plot_reduced(
    res: Result,
    ax=None,
    *,
    label: str | None = None,
    scale_to_max: bool = False,
    orientation: str = "h",
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
        # this should be length 1, sum just to be safe, is higher if in non-ROI mode
    for j, normed in enumerate(normalize_result(res, scale_to_max)):
        if label is not None:
            label = f"{label} ({j})"
        mask = np.isfinite(normed)
        if orientation == "h":
            ax.plot(res.tth[mask], normed[mask], label=label, **kwargs)
        elif orientation == "v":
            ax.plot(normed[mask], res.tth[mask], label=label, **kwargs)


def plot_ref(df, ax, scale_to_max=True, **kwargs):
    x = df["theta"]
    y = df["I1"]
    if scale_to_max:
        y /= y.max()

    ax.plot(x, y, **{"label": "reference", "scalex": False, **kwargs})


def raw_grid(df, results):
    fig = plt.figure(layout="constrained", figsize=(15, 15))
    grid_size = int(np.ceil(np.sqrt(len(df))))
    label_keys = set(find_varied_config([results[k][1] for k in df.index])) - {
        ("scan", "stop"),
        ("scan", "delta"),
    }
    df = df.sort_values(by=[".".join(lk) for lk in label_keys])
    sub_figs = fig.subfigures(grid_size, grid_size)
    for k, sfig in zip(df.index, sub_figs.ravel(), strict=False):
        tth, block = load_data(k)
        label = ", ".join(
            f"{_k}: {df.loc[k][_k]}" for _k in (".".join(_) for _ in label_keys)
        )
        plot_raw(tth, block[:, 0, 0, :], label, fig=sfig)
