# %% [markdown]
# proto-typing for building "database" of results

# %%

import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multianalyzer import Result
from scipy.interpolate import splrep, sproot

from bad_tools.config import (
    AnalyzerConfig,
    CompleteConfig,
)
from bad_tools.sim_reduction import (
    find_varied_config,
    load_all_config,
    plot_reduced,
    plot_ref,
    reduce_file,
)

# %%
os.environ["PYOPENCL_CTX"] = "1"

# %%
reference_pattern = pd.read_csv(
    "/home/tcaswell/Downloads/11bmb_7871_Y1.xye",
    skiprows=3,
    names=["theta", "I1", "I0"],
    sep=" ",
    skipinitialspace=True,
    index_col=False,
)

# %%

config = load_all_config(Path("/nsls2/data3/projects/next_iiia_hrd/xrt_output/"))
# config = load_all_config(Path("/mnt/scratch/hrd/sims/xrt_output/20250123"))

# %%
df = pd.DataFrame(
    {
        k: {
            f"{outer}.{inner}": v
            for outer, cfg in asdict(v).items()
            for inner, v in cfg.items()
        }
        for k, v in config.items()
    }
).T
df["job"] = [_.name.partition("-")[0] for _ in df.index]


# %%
for n, g in df.groupby("job"):
    print(n, len(g.index))
# %%

start = time.monotonic()
results = {k: reduce_file(k, phi_max=15) for k in config}
print(f"time = {time.monotonic() - start}")


# %%


def peak_fwha(tth, normed, limits: tuple[float, float], k: int = 10):
    slc = slice(np.searchsorted(tth, limits))
    tth = tth[slc]
    normed = normed[slc]
    half_max = np.max(normed) / 2.0
    s = splrep(tth, normed - half_max, k=k)
    return np.diff(sproot(s))


# %%


def plot_df(
    g: pd.DataFrame,
    results: dict[Path, tuple[Result, CompleteConfig, AnalyzerConfig]],
    title=None,
    reference_pattern=None,
):
    fig, ax = plt.subplots(layout="constrained")
    if title:
        fig.suptitle(title)
    label_keys = find_varied_config([results[k][1] for k in g.index])
    unit_convert: dict[tuple[str, str], tuple[Callable[float, float], str]] = (
        defaultdict(lambda: (lambda x: x, ""))
    )
    unit_convert.update(
        {
            ("source", "h_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
            ("source", "v_div"): (lambda x: np.deg2rad(x) * 1e3, "mrad"),
        }
    )
    g = g.sort_values(by=[".".join(lk) for lk in label_keys])
    for k in g.index:
        res, config, _ = results[k]
        label = " ".join(
            f"{sec}.{parm}={unit_convert[(sec, parm)][0](getattr(getattr(config, sec), parm)):.3g} {unit_convert[(sec, parm)][1]}"
            for sec, parm in label_keys
        )
        plot_reduced(res, ax=ax, scale_to_max=True, label=label, alpha=0.5)
    if reference_pattern is not None:
        plot_ref(reference_pattern, ax, scalex=False, linestyle=":")
    ax.set_xlim(7.5, 8.3)
    ax.legend()


plot_df(df[df["source.v_div"] == 0], results, "with vdiv=0")
plot_df(df[df["source.h_div"] == 0], results, "with hdiv=0")

# %%

for _, g in df.groupby("job"):
    plot_df(g, results)

plt.show()
