# %%

import matplotlib.pyplot as plt
import numpy as np
import xrt.backends.raycing.run as rrun

from hrd_tools.config import SimConfig, SourceConfig
from hrd_tools.xrt.hemisphere import Endstation, StripDetectorConfig

# %%
detector_config = StripDetectorConfig(
    radius=1_000, strip_width=0.05, strip_height=8, center=(0, 0, 0)
)
sim_config = SimConfig(nrays=100_000)

source_config = SourceConfig(
    E_incident=29_400,
    pattern_path="/home/tcaswell/Downloads/11bmb_7871_Y1.xye",
    dx=1,
    dz=0.1,
    dy=0,
    delta_phi=1,
    E_hwhm=1.4e-4,
    max_tth=90,
    min_tth=0,
)


# %%
endstation = Endstation.from_configs(source_config, detector_config, sim_config)


# %%
def show_bl(endstation: Endstation):
    def run_process(beamLine):
        # "raw" beam
        geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()
        screen01beamLocal01 = beamLine.screen_main.expose(
            beam=geometricSource01beamGlobal01
        )
        screen02beamLocal01 = beamLine.strip_sphere.expose(
            beam=geometricSource01beamGlobal01
        )
        outDict = {
            "source": geometricSource01beamGlobal01,
            "source_screen": screen01beamLocal01,
            "strip_screen": screen02beamLocal01,
        }

        beamLine.prepare_flow()
        return outDict

    rrun.run_process = run_process
    endstation.bl.glow()


# %%
show_bl(endstation)


# %%


def build_hist(endstation, lb, *, isScreen=True):
    good = (lb.state == 1) | (lb.state == 2)
    if isScreen:
        x, y, z = lb.x[good], lb.z[good], lb.y[good]
    else:
        x, y, z = lb.x[good], lb.y[good], lb.z[good]

    flux = lb.Jss[good] + lb.Jpp[good]

    # locations in local coordinates, re-mapped
    # x is beam direction (y in xrt beam line view)
    # y is xrt beamline x (inboard/outboard)
    # z is xrt beamline z (up/down)
    hit_locations = np.vstack([x, y, z]).T

    # estimate which rays would hit a flat detector of a given height, this
    # marginally over estimates rays that intersect the sphere at the edges but
    # would miss the strip.  Given the size of the sphere (~1m) vs the size of
    # the detector (~10mm) this OK to first order
    # assume the detector is centered on the meridian
    on_strip = np.abs(hit_locations[:, 1]) < endstation.detector.strip_height / 2
    # only care about y (beam direction), z (up) now
    hit_locations = hit_locations[on_strip]
    flux = flux[on_strip]
    th = np.arctan2(hit_locations[:, 2], hit_locations[:, 0])
    dth = np.arcsin(endstation.detector.strip_width / endstation.detector.radius)
    print(len(th))
    # slightly below direct beam to "up"
    bins = np.arange(-np.pi / 8, np.pi / 2 - np.pi / 8, dth)

    hist, bins = np.histogram(th, bins=bins)  # , weights=flux)

    return hist, bins


# %%


def oro(endstation):
    import matplotlib.pyplot as plt

    (a,) = endstation.run_process().values()
    x, y, z = a.x, a.z, a.y
    hit_locations = np.vstack([x, y, z]).T

    fig, ax_list = plt.subplots(3, layout="constrained")

    for label, data, ax in zip("xyz", hit_locations.T, ax_list, strict=True):
        ax.hist(data, bins="auto")
        ax.set_title(label)


# %%
def batch_run(endstation, N=100):
    I_stack = []
    for j in range(N):
        (a,) = endstation.run_process().values()
        I, bins = build_hist(endstation, a)
        I_stack.append(I)
    return bins[:-1] + np.diff(bins), np.vstack(I_stack)


def scan_position(offsets, *, N=10, delta_phi=1):
    sim_config = SimConfig(nrays=1_000_000)

    source_config = SourceConfig(
        E_incident=29_400,
        pattern_path="/home/tcaswell/Downloads/11bmb_7871_Y1.xye",
        dx=1,
        dz=0.1,
        dy=0,
        delta_phi=delta_phi,
        E_hwhm=1.4e-4,
        max_tth=90,
        min_tth=0,
    )

    out_stack = []
    for offset in offsets:
        detector_config = StripDetectorConfig(
            radius=1_000, strip_width=0.05, strip_height=8, center=offset
        )

        endstation = Endstation.from_configs(source_config, detector_config, sim_config)
        th, I = batch_run(endstation, N=10)
        out_stack.append(I.mean(axis=0))

    return th, np.vstack(out_stack)


# %%

th, scan_img4 = scan_position(
    [(j, 0, 0) for j in np.arange(0, 200, 5)], delta_phi=90, N=50
)

# %%

fig, ax = plt.subplots(layout="constrained", figsize=(6, 2))
im = ax.imshow(
    scan_img4,
    aspect="auto",
    norm="log",
    extent=[np.rad2deg(th.min()), np.rad2deg(th.max()), 0, 200],
    origin="lower",
)
fig.colorbar(im)
ax.set_xlabel(r"$2\theta$ (deg) on detector")
ax.set_ylabel("outboard offset (mm)")
ax.set_xlim(0, 55)
plt.show()
