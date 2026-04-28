"""Gauge-volume edges relative to a single pixel (stub)."""

# %%
import matplotlib.pyplot as plt
import numpy as np
import xrt.backends.raycing.materials as rmats

import _fdr_params

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)
_blessed = _fdr_params.complete_config()

E = (_args.energy_keV if _args.energy_keV is not None
     else _blessed.source.E_incident / 1000.0) * 1000.0       # eV
crystal = rmats.CrystalSi(t=_fdr_params.crystal_reference()["thickness_mm"])
bragg = crystal.get_Bragg_angle(E)
# in radians
darwin = crystal.get_Darwin_width(E)

pixel_size = _fdr_params.detector().pixel_pitch              # µm

# %%

fig, ax = plt.subplots()

ax.axline((0, pixel_size / 2), slope=np.tan(darwin))
ax.axline((0, -pixel_size / 2), slope=-np.tan(darwin))

ax.set_xlim(0, 1.1e6)
_save(fig, "gauge_volume.png")
_fdr_params.maybe_show(_args)
