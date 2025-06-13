# %%
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import xrt.backends.raycing.materials as rmats

# %%
def angle(n1, v1):
    return np.arccos(np.linalg.vecdot(n1, v1) / (np.linalg.vector_norm(n1) * np.linalg.vector_norm(v1, axis=1)))

def to_norm(θ, ɸ):
    return np.array([np.sin(θ) * np.cos(ɸ),
                     np.sin(θ) * np.sin(ɸ),
                     np.cos(θ) * np.ones_like(ɸ)
                     ])


# %%
E = 30_000
crystal = rmats.CrystalSi(t=1)
bragg = crystal.get_Bragg_angle(E)
darwin = crystal.get_Darwin_width(E)
phi = np.deg2rad(np.linspace(0, 1, 128))

# %%

fig, ax = plt.subplots()

for arm in np.deg2rad([0, 45, 90]):
    n1 = to_norm(arm - bragg - np.pi/2, 0)
    v = to_norm(arm, phi).T

    ax.plot(np.rad2deg(phi), np.rad2deg(bragg - (angle(n1, v) - np.pi/2)),
            label=np.rad2deg(arm))

ax.axhline(np.rad2deg(darwin)    )
ax.set_xlabel('$\phi$ (deg)')
ax.set_ylabel(r'$\theta_{bragg} - 2\theta_{eff}$ (deg)')
ax.legend()

# %%

out = []
arm_th = np.linspace(-90, 180, 512)
for arm in np.deg2rad(arm_th):
    n1 = to_norm(arm - bragg - np.pi/2, 0)
    v = to_norm(arm, phi).T
    out.append(np.rad2deg(bragg - (angle(n1, v) - np.pi/2)))

plt.imshow(out, aspect='auto', extent=[np.rad2deg(np.min(phi)), np.rad2deg(np.max(phi)), np.min(arm_th), np.max(arm_th)], origin='lower')
plt.colorbar()
