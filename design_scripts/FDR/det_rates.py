"""Print required scan speed and detector frame-rate table.

Run with::

    pixi run -e xrt python design_scripts/FDR/det_rates.py
"""

from itertools import product

import numpy as np

import _fdr_params

_args = _fdr_params.parse_args(__doc__)

dt = _fdr_params.layout()["frame_dt_s"]   # s

motion = [5, 10, 15, 20, 25]              # deg
target_time = [30, 60]                    # s

print("dist\ttime\tspeed\tframe rate")
for time, dist in product(target_time, motion):
    print(f"{dist}\t{time}\t{dist / time:.2f}\t{int(np.ceil((dist / time) / dt))}")
