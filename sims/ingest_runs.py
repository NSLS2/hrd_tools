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


from hrd_tools.config import (
    AnalyzerConfig,
    CompleteConfig,
    AnalyzerCalibration,
    SimScanConfig,
)
from hrd_tools.sim_reduction import (
    find_varied_config,
    plot_reduced,
    plot_ref,
    reduce_file,
    plot_raw,
    load_data,
    reduce_raw,
    reduce_catalog,
    load_reference,
    plot_reduced_cat,
)
from hrd_tools.file_io import load_all_config, dflt_config
from hrd_tools.sim_report import generate_report, Reduced

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
os.environ["PYOPENCL_CTX"] = "0"


# %%


reference_pattern = load_reference(
    "/nsls2/data3/projects/next_iiia_hrd/sim_input/11bmb_7871_Y1.xye"
)

# %%
c_roll = c.search(tq.Contains("varied_key", "analyzer.roll"))

# %%
c_roll


# %%


# from sim_reduction import

# %%
# out = {}
# for j in range(5):
#     out.update(reduce_catalog(c.values_indexer[-(j + 1)], phi_max=5))


# %%


# plot_reduced_cat(c_roll.values_indexer[-1], out)


# %%
# plot_reduced_cat(c.values_indexer[-3], out, reference_pattern=reference_pattern)

# %%
c2 = c_roll.items_indexer[1][1]
out2 = reduce_catalog(c2, phi_max=5)

# %%
plot_reduced_cat(c2, out2)


# %%
plot_reduced_cat(c2, out2, reference_pattern=reference_pattern)

# %%
from hrd_tools.sim_reduction import normalize_result


# fig = raw_grid(c.values_indexer[-3], out)
# %%


# %%

# %%


# %%

plot_cat_fwhm_1d(c2, out2, [(7.65, 7.72), (7.975, 8.1)])

# %%
from typing import Any


# %%

# %%

r = Reduced.from_cat(c_roll.items_indexer[1][1], 45)


# %%


# %%

# r = Reduced.from_cat(c.values_indexer[-3], 45)
generate_report(r, "bob.md")
# %%
for k, v in c.items():
    r = Reduced.from_cat(v, 45)
    generate_report(r, f"{k}.md")

# %%
