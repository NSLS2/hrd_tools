# %% [markdown]
#
# The goal here is to bench mark doing simple summing on chunked data using
# data we took at APS as a representative case.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import sparse
import tqdm
from multihead.config import DetectorROIs
from multihead.file_io import open_data

import _fdr_params

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)

# %%
rois = DetectorROIs.from_yaml(_fdr_params.rois_path())
roi_slices = [r.to_slices() for r in rois.rois.values()]

# %%
_rd = _fdr_params.real_data()
root = Path(_rd["root"])
raw = open_data(root / _rd["dataset"], _rd["version"])
all_data = raw._sparse_data
arm_thetas = raw.get_arm_tth()

# %%
batch_size = 1000
batches = []
for j in tqdm.tqdm(range(1 + all_data.shape[1] // batch_size)):
    batch = all_data[:, j * batch_size : (j + 1) * batch_size]
    batches.append(
        pa.Table.from_arrays(
            [*batch.coords, batch.data],
            names=("detector", "frame", "row", "col", "data"),
            metadata={"shape": json.dumps(batch.shape)},
        )
    )
arm_batches = [
    arm_thetas[j * batch_size : (j + 1) * batch_size]
    for j in range(1 + all_data.shape[1] // batch_size)
]

# %%
output = np.zeros((12, all_data.shape[1])) * np.nan
thetas = np.zeros_like(output) + np.arange(12).reshape(-1, 1) * 2
fig, ax = plt.subplots()
lines = ax.plot(output.T)
ax.set_ylim(0, 120_000)
ax.set_xlim(0, 55)
for j, (b, th) in tqdm.tqdm(enumerate(zip(batches, arm_batches, strict=True))):
    start_frame = np.min(b["frame"])
    end_frame = np.max(b["frame"])
    patch_slc = slice(j * batch_size + start_frame, j * batch_size + end_frame + 1)
    thetas[:, patch_slc] += th
    new_frames = sparse.COO(
        [b[k] for k in ["detector", "frame", "row", "col"]],
        data=b["data"],
        shape=(12, end_frame + 1, 256, 256),
    )
    for k, (rslc, cslc) in enumerate(roi_slices):
        output[k, patch_slc] = new_frames[k, :, rslc, cslc].sum(axis=(1, 2)).todense()

    for k, (ln, o, lth) in enumerate(zip(lines, output, thetas, strict=True)):
        ln.set_data(lth, o + k * 1_000)

    plt.pause(0.00001)

_save(fig, "live_data_validation.png")
_fdr_params.maybe_show(_args, block=True)
