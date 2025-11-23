# %%
import matplotlib.pyplot as plt
import numpy as np



def intercept(tth, w, d):
    tth = np.deg2rad(tth)
    w = np.deg2rad(w)
    return np.sin(np.pi - w) * d / np.sin(w + tth - np.pi)

d = 2
tth = np.linspace(-10, 120, 1024)

# %% [markdown]
#
# If we do not consider the extended nature of the gague
# volume we get divergences

# %%
fig, ax = plt.subplots(layout='constrained')

for w in np.linspace(1, 179, 5):
    ax.plot(tth, intercept(tth, w, d), label=f'{w=}')
ax.set_ylim(-1.5*d, 1.5*d)
ax.legend()
plt.show()
