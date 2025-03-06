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

from hrd_tools.config import AnalyzerConfig, CompleteConfig, AnalyzerCalibration
from hrd_tools.sim_reduction import (
    find_varied_config,
    plot_reduced,
    plot_ref,
    reduce_file,
    plot_raw,
    load_data,
    raw_grid,
    reduce_raw,
)
from hrd_tools.file_io import load_all_config, dflt_config

# %%
from tiled.client import from_uri
import tiled.queries as tq
import tiled

# %%
plt.plot(range(5))

# %%
c = from_uri("http://localhost:8000")["raw"]

# %%
list(c)

# %%
import sys

# %%
sys.executable

# %%
os.environ["PYOPENCL_CTX"] = "0"


# %%
import functools


@functools.lru_cache
def load_reference(fname):
    return pd.read_csv(
        fname,
        skiprows=3,
        names=["theta", "I1", "I0"],
        sep=" ",
        skipinitialspace=True,
        index_col=False,
    )


reference_pattern = load_reference(
    "/nsls2/data3/projects/next_iiia_hrd/sim_input/11bmb_7871_Y1.xye"
)

# %%
c_roll = c.search(tq.Contains("varied_key", "analyzer.roll"))

# %%
c_roll


# %%
def load_config_from_tiled(grp: tiled.client.container.Container):
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
        tth = grp["tth"].read()
        configs["scan"] = SimScanConfig(
            start=np.rad2deg(tth[0]),
            stop=np.rad2deg(tth[-1]),
            delta=np.rad2deg(np.mean(np.diff(tth))),
        )
    return CompleteConfig(**configs)


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

        tth = np.rad2deg(sim["tth"].read())
        channels = sim["block"].read().astype("int32")
        # for j in range(config.analyzer.N):
        #     plot_raw(tth, channels[:, j, :, :].squeeze(), f"crystal {j}")
        ret = reduce_raw(
            block=channels,
            tths=tth,
            tth_min=np.float64(config.scan.start),
            tth_max=config.scan.stop
            + (config.analyzer.N - 1) * np.rad2deg(config.analyzer.cry_offset),
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
# out = {}
# for j in range(5):
#     out.update(reduce_catalog(c.values_indexer[-(j + 1)], phi_max=5))


# %%
def plot_reduced_cat(
    cat,
    results: dict[Path, tuple[Result, CompleteConfig, AnalyzerConfig]],
    title=None,
    reference_pattern=True,
    *,
    config_filter=None,
):
    fig, ax = plt.subplots(layout="constrained")
    if title:
        fig.suptitle(title)
    label_keys = tuple(_.split(".") for _ in cat.metadata["varied_key"])
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
        if config_filter is not None and not config_filter(run.metadata):
            continue
        (res, config, _) = results[(cat.uri, k)]
        label = " ".join(
            f"{sec}.{parm}={unit_convert[(sec, parm)][0](getattr(getattr(config, sec), parm)):.3g} {unit_convert[(sec, parm)][1]}"
            for sec, parm in label_keys
        )
        plot_reduced(res, ax=ax, scale_to_max=True, label=label, alpha=0.5)
    if reference_pattern:
        reference_data = load_reference(
            cat.metadata["static_values"]["source"]["pattern_path"]
        )
        plot_ref(reference_data, ax, scalex=False, linestyle=":")
    ax.set_xlim(7.5, 8.3)
    ax.legend()
    return fig


# plot_reduced_cat(c_roll.values_indexer[-1], out)


# %%
# plot_reduced_cat(c.values_indexer[-3], out, reference_pattern=reference_pattern)

# %%
c2 = c_roll.items_indexer[0][1]
out2 = reduce_catalog(c2, phi_max=5)

# %%
plot_reduced_cat(c2, out2)


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


def raw_grid(cat, results, config_filter=None):
    fig = plt.figure(layout="constrained", figsize=(15, 15))
    grid_size = int(np.ceil(np.sqrt(len(cat))))
    label_keys = cat.metadata["varied_key"]
    unit_convert: dict[tuple[str, str], tuple[Callable[float, float], str]] = (
        defaultdict(lambda: (lambda x: x, ""))
    )
    unit_convert.update(
        {
            ("source", "h_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
            ("source", "v_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
        }
    )
    sub_figs = fig.subfigures(grid_size, grid_size - 1)
    for (k, sim), sfig in zip(cat.items(), sub_figs.ravel(), strict=False):
        tth = np.rad2deg(sim["tth"].read())
        block = sim["block"].read().astype("int32")
        (res, config, _) = results[(cat.uri, k)]
        label = " ".join(
            f"{sec}.{parm}={unit_convert[(sec, parm)][0](getattr(getattr(config, sec), parm)):.3g} {unit_convert[(sec, parm)][1]}"
            for sec, parm in (_.split(".") for _ in label_keys)
        )
        axd = plot_raw(tth, block[:, 0, 0, :], label, fig=sfig)
        plot_reduced(res, ax=axd["parasite"], orientation="v", scale_to_max=True)

    return fig


# fig = raw_grid(c.values_indexer[-3], out)
# %%
import io
import base64


def base64ify(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


# %%
def peak_fwhm(tth, normed, limits: tuple[float, float]):
    slc = slice(*np.searchsorted(tth, limits))
    tth = tth[slc]
    normed = normed[slc]
    half_max = np.max(normed) / 2.0
    s = splrep(tth, normed - half_max, k=3)
    return np.diff(sproot(s))


# %%


def cat_to_fwhm(
    cat,
    results: dict[Path, tuple[Result, CompleteConfig, AnalyzerConfig]],
    limits: tuple[float, float],
    *,
    config_filter=None,
):
    ret = {}
    for key, run in cat.items():
        if config_filter is not None and not config_filter(run.metadata):
            continue
        (res, config, _) = results[(cat.uri, key)]
        tth = res.tth
        (normed,) = normalize_result(res)
        fwhm = peak_fwhm(tth, normed, limits=limits)
        if len(fwhm) > 1:
            continue
        ret[key] = float(fwhm)

    reference_data = load_reference(
        cat.metadata["static_values"]["source"]["pattern_path"]
    )

    return ret, peak_fwhm(reference_data["theta"], reference_data["I1"], limits=limits)


# %%


def plot_cat_fwhm_1d(cat, results, peak_windows: list[tuple[float, float]]):
    fwhm = []
    for limits in peak_windows:
        fwhm.append(cat_to_fwhm(cat, results, limits))

    # flatten the varied values
    varied = {
        k: val
        for k, v in cat.metadata["varied_values"].items()
        for _, vv in v.items()
        for _, val in vv.items()
    }
    print(varied)

    fig, ax = plt.subplots(layout="constrained")
    for peak_fwhm, ref_peak in fwhm:
        x = [varied[k] for k in peak_fwhm]
        y = list(peak_fwhm.values())
        ax.plot(x, y, "o")
        ax.axhline(ref_peak, linestyle="--")
    return fig, fwhm


# %%

plot_cat_fwhm_1d(c2, out2, [(7.9, 8.1)])

# %%
from typing import Any


def aggregate_min_max(
    data: list[dict[str, dict[str, Any]]],
) -> dict[str, tuple[Any, Any]]:
    """
    Aggregate a list of nested dictionaries into a single dictionary.

    Each key in the output is formed by concatenating the outer and inner keys with a dot.
    The corresponding value is a tuple (min, max) representing the minimum and maximum values
    found for that inner key across all dictionaries.

    Parameters
    ----------
    data : list of dict[str, dict[str, Any]]
        A list of dictionaries where each dictionary maps an outer key (str) to an inner
        dictionary (dict[str, Any]). The inner dictionary's values should be comparable
        (e.g., numbers).

    Returns
    -------
    dict[str, tuple[Any, Any]]
        A dictionary with keys in the format "outer.inner" and values as tuples (min, max)
        computed from the aggregated inner dictionary values.

    Examples
    --------
    >>> data = [
    ...     {"A": {"x": 10, "y": 20}},
    ...     {"A": {"x": 15, "y": 25}},
    ...     {"B": {"z": 5}},
    ...     {"B": {"z": 7, "w": 3}}
    ... ]
    >>> aggregate_min_max(data)
    {'A.x': (10, 15), 'A.y': (20, 25), 'B.z': (5, 7), 'B.w': (3, 3)}
    """
    result: dict[str, tuple[Any, Any]] = {}

    for outer_dict in data:
        for outer_key, inner_dict in outer_dict.items():
            for inner_key, value in inner_dict.items():
                composite_key = f"{outer_key}.{inner_key}"
                if composite_key not in result:
                    result[composite_key] = (value, value)
                else:
                    current_min, current_max = result[composite_key]
                    result[composite_key] = (
                        min(current_min, value),
                        max(current_max, value),
                    )

    return result


# %%

from jinja2 import Template

# Define the updated Jinja2 template as a multi-line string
template_str = """
# {{ title }}

{{ description }}
## Scanned Parameters

{% if scanned_parameters %}
{% for param in scanned_parameters %}
- **{{ param.name }}:** {{ param.range }}
{% endfor %}
{% else %}
*No parameters were scanned.*
{% endif %}

## static Parameters

{% if static_parameters %}
{% for section, table in static_parameters.items() %}
### {{ section }}

| Key | Value |
| --- | ----- |{% for key, value in table.items() %}
| {{ key }} | {{ value }} |{% endfor %}

{% endfor %}
{% endif %}


## Graphs

{% if images %}
{% for image in images %}
![{{ image.caption }}]({{ image.filename }})

*{{ image.caption }}*
{% endfor %}
{% else %}
*No graphs available.*
{% endif %}
"""

# Create a Template object from the string
template = Template(template_str)


# %%
def generate_report(cat, integrations):
    data = {
        "title": "Simulation Report",
        "description": cat.metadata["short_descrption"],
        "static_parameters": cat.metadata["static_values"],
        "scanned_parameters": [
            {"name": k, "range": f"{vmin} - {vmax}"}
            for k, (vmin, vmax) in aggregate_min_max(
                cat.metadata["varied_values"].values()
            ).items()
        ],
        "images": [
            {
                "caption": "integrations",
                "filename": f"data:image/png;base64,{base64ify(plot_reduced_cat(cat, integrations, reference_pattern=True))}",
            },
            # {"caption": "Full Scans", "filename": f"data:image/png;base64,{base64ify(raw_grid(cat, integrations))}"},
        ],
    }

    with open("test.md", "w") as fout:
        fout.write(template.render(data))


generate_report(c2, out2)
# %%
c2["1196396"]["0"].metadata
# %%
