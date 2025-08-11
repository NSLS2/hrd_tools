import bluesky.plans as bp
import numpy as np
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

from hrd_tools.sim import make_sample_rack

rng = np.random.default_rng(12345)

bec = BestEffortCallback()
bec.disable_table()

x, y, det = make_sample_rack(3 * rng.normal(size=15) + 13, 5)

det.kind = "hinted"

RE = RunEngine()
RE.subscribe(bec)

RE(bp.grid_scan([det], y, 5, 20, 15, x, 1, 15, 200))
