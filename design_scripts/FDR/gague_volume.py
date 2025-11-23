# %%
import matplotlib.pyplot as plt
import numpy as np
import xrt.backends.raycing.materials as rmats

E = 25_000
crystal = rmats.CrystalSi(t=1)
bragg = crystal.get_Bragg_angle(E)
# in radians
darwin = crystal.get_Darwin_width(E)

pixel_size = 55

# %%

fig, ax = plt.subplots()

ax.axline((0, pixel_size / 2), slope=np.tan(darwin))
ax.axline((0, -pixel_size / 2), slope=-np.tan(darwin))

ax.set_xlim(0, 1.1e6)
