# %% [markdown]
#
# The goal here is to bench mark doing simple summing on chunked data using
# data we took at APS as a representative case

# %%
import json
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import sparse
import tqdm
from multihead.config import DetectorROIs
from multihead.file_io import open_data

# %%
roi_data = StringIO("""rois:
- detector_number: 1
  roi_bounds:
    cslc:
    - 0
    - 208
    rslc:
    - 120
    - 136
- detector_number: 2
  roi_bounds:
    cslc:
    - 25
    - 240
    rslc:
    - 77
    - 94
- detector_number: 3
  roi_bounds:
    cslc:
    - 11
    - 219
    rslc:
    - 176
    - 194
- detector_number: 4
  roi_bounds:
    cslc:
    - 25
    - 236
    rslc:
    - 188
    - 204
- detector_number: 5
  roi_bounds:
    cslc:
    - 12
    - 221
    rslc:
    - 185
    - 202
- detector_number: 6
  roi_bounds:
    cslc:
    - 30
    - 240
    rslc:
    - 122
    - 139
- detector_number: 7
  roi_bounds:
    cslc:
    - 15
    - 220
    rslc:
    - 162
    - 179
- detector_number: 8
  roi_bounds:
    cslc:
    - 30
    - 238
    rslc:
    - 70
    - 87
- detector_number: 9
  roi_bounds:
    cslc:
    - 13
    - 217
    rslc:
    - 166
    - 183
- detector_number: 10
  roi_bounds:
    cslc:
    - 32
    - 238
    rslc:
    - 115
    - 132
- detector_number: 11
  roi_bounds:
    cslc:
    - 0
    - 211
    rslc:
    - 122
    - 139
- detector_number: 12
  roi_bounds:
    cslc:
    - 36
    - 237
    rslc:
    - 59
    - 78""")

rois = DetectorROIs.from_yaml(roi_data)
roi_slices = [r.to_slices() for r in rois.rois.values()]
# %%

root = Path("/home/tcaswell/data/hrd/aps/Oct25/raw/v3/")
# the full data set
raw = open_data(root / "11bmb_2387_mda_defROI", 3)
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
