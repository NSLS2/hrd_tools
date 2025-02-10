from random import randint

import matplotlib.pyplot as plt
import numpy as np

from hrd_tools.search import LocRange, Sample, SampleLocation, SampleViz

N = 15


viz_list = []

for j, name in enumerate([f"s{k:02d}" for k in range(N)]):
    sample = Sample(name, {})
    sample_loc = SampleLocation(
        sample, LocRange(2 * j + 1, 0.5), LocRange(randint(8, 15), randint(5, 9))
    )
    viz = SampleViz(sample_loc)

    viz_list.append(viz)


fig, ax = plt.subplots()

ax.hlines([1, 24], 0, N * 2, lw=5, color=".5", zorder=-1)
ax.vlines(np.arange(1, N * 2 + 1, 2), 1, 24, color=".9", lw=10, zorder=-2)

for viz in viz_list:
    viz.show_box(ax, callback=lambda sample: print(f"clicked on {sample.name}"))


ax.set_xlim(-1, N * 2 + 1)
ax.set_ylim(0, 25)

plt.show()
