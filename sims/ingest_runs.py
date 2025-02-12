# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext,pixi-kernel
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (Pixi - Advanced)
#     language: python
#     name: pixi-kernel-python3
#   pixi-kernel:
#     environment: default
# ---

# %%
# %matplotlib widget


# %%
 
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multianalyzer import Result
from scipy.interpolate import splrep, sproot

from hrd_tools.config import (
    AnalyzerConfig,
    CompleteConfig,
    AnalyzerCalibration
)
from hrd_tools.sim_reduction import (
    find_varied_config,
    
    plot_reduced,
    plot_ref,
    reduce_file,
    plot_raw,
    load_data,
    raw_grid,
reduce_raw
)
from hrd_tools.file_io import load_all_config, dflt_config

# %%
from tiled.client import from_uri
import tiled.queries as tq
import tiled

# %%
plt.plot(range(5))

# %%
c = from_uri('http://localhost:8000')['raw']

# %%
list(c)

# %%
import sys

# %%
sys.executable

# %%
os.environ["PYOPENCL_CTX"] = "0"

# %%
reference_pattern = pd.read_csv(
    "/nsls2/data3/projects/next_iiia_hrd/sim_input/11bmb_7871_Y1.xye",
    skiprows=3,
    names=["theta", "I1", "I0"],
    sep=" ",
    skipinitialspace=True,
    index_col=False,
)

# %%
c_roll = c.search(tq.Contains('varied_key', 'analyzer.roll'))

# %%
c_roll


# %%
def load_config_from_tiled(grp:tiled.client.container.Container ):
    configs = {}
    md = grp.metadata
    for fld in fields(CompleteConfig):
        try:
            config_grp = md[fld.name]
        except KeyError:
            if fld.name != "scan":
                print(f"missing {fld.name}")
        else:
            configs[fld.name] = fld.type(**config_grp)
    if "scan" not in configs:
        tth = grp['tth'].read()
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
   

def reduce_catalog(
    cat: tiled.client.container.Container,
    calib: AnalyzerCalibration | None = None,
    mode: str = "opencl",
    dtth_scale: float = 5.0,
    phi_max: float = 90,
):
    out = {}
    for k, sim in cat.items():
        
        config = load_config_from_tiled(sim)
        if calib is None:
            calib_sim = dflt_config(config)
        else:
            calib_sim = calib
        dtth = config.scan.delta * dtth_scale
        
        
        tth = np.rad2deg(sim['tth'].read())
        channels = sim['block'].read().astype("int32")
        # for j in range(config.analyzer.N):
        #     plot_raw(tth, channels[:, j, :, :].squeeze(), f"crystal {j}")
        ret = reduce_raw(
        block=channels,
        tths=tth,
        tth_min=np.float64(config.scan.start),
        tth_max=config.scan.stop + (config.analyzer.N -1) * np.rad2deg(config.analyzer.cry_offset),
        dtth=dtth,
        analyzer=config.analyzer,
        detector=config.detector,
        calibration=calib_sim,
        phi_max=np.float64(phi_max),
        mode=mode,
        width=0,
    )
        out[(cat.uri, k)] = (ret, config, calib)
        
    # fig, ax = plt.subplots()
    # for j in range(config.analyzer.N):
    #     break
    #     ax.plot(ret.tth, 0.1 * j + ret.signal[j, :, 0] / ret.norm[j], label=j)
    # ax.legend()
    return out

    


# %%
out = {}
for j in range(5):
    out.update(reduce_catalog(c.values_indexer[-(j+1)], phi_max=5))
    

# %%
def plot_reduced_cat(
    cat,
    results: dict[Path, tuple[Result, CompleteConfig, AnalyzerConfig]],
    
    title=None,
    reference_pattern=None,
    *,
    config_filter = None
):
    fig, ax = plt.subplots(layout="constrained")
    if title:
        fig.suptitle(title)
    label_keys = tuple(_.split('.') for _ in cat.metadata['varied_key'])
    print(label_keys)
    unit_convert: dict[tuple[str, str], tuple[Callable[float, float], str]] = (
        defaultdict(lambda: (lambda x: x, ""))
    )
    unit_convert.update(
        {
            ("source", "h_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
            ("source", "v_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
        }
    )
    for k, run in cat.items():
        if config_filter is not None:
            if not config_filter(run.metadata):
                continue
        (res, config, _) = results[(cat.uri, k)]
        label = " ".join(
            f"{sec}.{parm}={unit_convert[(sec, parm)][0](getattr(getattr(config, sec), parm)):.3g} {unit_convert[(sec, parm)][1]}"
            for sec, parm in label_keys
        )
        plot_reduced(res, ax=ax, scale_to_max=True, label=label, alpha=0.5)
    if reference_pattern is not None:
        plot_ref(reference_pattern, ax, scalex=False, linestyle=":")
    # ax.set_xlim(7.5, 8.3)
    ax.legend()

# plot_reduced_cat(c_roll.values_indexer[-1], out)


# %%
plot_reduced_cat(c.values_indexer[-3], out, reference_pattern=reference_pattern)

# %%
c2 = c['1197776']
out2 = reduce_catalog(c2, phi_max=90)
#plot_reduced_cat(c2, out)

# %%
def peak_fwha(tth, normed, limits: tuple[float, float], k: int = 10):
    slc = slice(np.searchsorted(tth, limits))
    tth = tth[slc]
    normed = normed[slc]
    half_max = np.max(normed) / 2.0
    s = splrep(tth, normed - half_max, k=k)
    return np.diff(sproot(s))


# %%
plot_reduced_cat(c2, out2, reference_pattern=reference_pattern)

# %%
from hrd_tools.sim_reduction import normalize_result

def plot_reduced(
    res: Result,
    ax=None,
    *,
    label: str | None = None,
    scale_to_max: bool = False,
    orientation: str = 'h',
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
        # this should be length 1, sum just to be safe, is higher if in non-ROI mode
    for j, normed in enumerate(normalize_result(res, scale_to_max)):
        if label is not None:
            label = f"{label} ({j})"
        mask = np.isfinite(normed)
        if orientation == 'h':
            ax.plot(res.tth[mask], normed[mask], label=label, **kwargs)
        elif orientation == 'v':
            ax.plot(normed[mask], res.tth[mask], label=label, **kwargs)


def raw_grid(cat, results, config_filter=None):
    fig = plt.figure(layout="constrained", figsize=(15, 15))
    grid_size = int(np.ceil(np.sqrt(len(cat))))
    label_keys = cat.metadata['varied_key']
    unit_convert: dict[tuple[str, str], tuple[Callable[float, float], str]] = (
        defaultdict(lambda: (lambda x: x, ""))
    )
    unit_convert.update(
        {
            ("source", "h_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
            ("source", "v_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
        }
    )
    sub_figs = fig.subfigures(grid_size, grid_size -1)
    for (k, sim), sfig in zip(cat.items(), sub_figs.ravel(), strict=False):
        tth = np.rad2deg(sim['tth'].read())
        block = sim['block'].read().astype("int32")
        (res, config, _) = results[(cat.uri, k)]
        label = " ".join(
            f"{sec}.{parm}={unit_convert[(sec, parm)][0](getattr(getattr(config, sec), parm)):.3g} {unit_convert[(sec, parm)][1]}"
            for sec, parm in (_.split('.') for _ in label_keys)
        )
        axd = plot_raw(tth, block[:, 0, 0, :], label, fig=sfig)
        plot_reduced(res, ax=axd['parasite'], orientation='v', scale_to_max=True)
        
raw_grid(c.values_indexer[-3], out)


# %%
def one_line_summary(g):
    label_keys = ['.'.join(lk) for lk in set(find_varied_config([results[k][1] for k in g.index]))]
    desc = g[label_keys].describe()
    print(desc)
for _, g in df.groupby("job"):
    one_line_summary(g)
    plot_df(g, results, reference_pattern=reference_pattern)
    
plt.show()

# %%
