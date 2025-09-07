import numpy as np
from itertools import product

dt = 1e-4

motion = [5, 10, 15, 20, 25]
target_time = [30, 60]

for time, dist in product(target_time, motion):
    print(f'{dist}, {time}, {dist/time:.2f}, {int(np.ceil((dist/time) / dt))}')
