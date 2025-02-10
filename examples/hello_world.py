import bluesky.plans as bp
import matplotlib.pyplot as plt
import numpy as np
from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback

from hrd_tools.sim import make_detector

bec = BestEffortCallback()

det = make_detector(np.linspace(0, np.pi / 2, 4), np.sin)


RE = RunEngine()
RE.subscribe(bec)

RE(bp.scan(det.channels, det.angle, 0, 2 * np.pi, 128))

plt.show(block=True)
