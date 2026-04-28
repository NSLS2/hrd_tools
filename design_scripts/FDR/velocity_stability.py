# %% [markdown]
# The goal here is to generate an estimate of how stable
# 11BM is to guide how stable we can request.

# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.signal
from matplotlib.patches import Rectangle
from multihead.config import DetectorROIs
from multihead.file_io import open_data

import _fdr_params

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)

# %%
_rd = _fdr_params.real_data()
root = Path('/mnt/store/bnl/cache/hrd/beamtime_cleanup/reorg/raw/v1/')
raw = open_data(root / '11bmb_2387', 1)

# %%

frac = np.ptp(raw._mda.detectors['Clock50Mhz'][5:]) / np.mean(raw._mda.detectors['Clock50Mhz'][5:])

print(f"The ptp / mean is {frac:.2f} at APS")
