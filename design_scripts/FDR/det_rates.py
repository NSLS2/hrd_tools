from itertools import product

import numpy as np

dt = 1e-4

motion = [5, 10, 15, 20, 25]
target_time = [30, 60]

print('dist\ttime\tspeed\tframe rate')
for time, dist in product(target_time, motion):
    print(f"{dist}\t{time}\t{dist / time:.2f}\t{int(np.ceil((dist / time) / dt))}")
