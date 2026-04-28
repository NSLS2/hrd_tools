# %% [markdown]
# # Detector layout
#
# This generates a schematic of the analyzer-arm crystal/detector layout
# (banks of crystals on a primary arm) and a few variants on a grid.

# %%
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

import _fdr_params

_args = _fdr_params.parse_args(__doc__)
_save = _fdr_params.figure_saver(_args)
_layout = _fdr_params.layout()


@dataclass
class Bank:
    number: int
    offset: float

    @property
    def solid_angle(self):
        return (self.number - 1) * self.offset


@dataclass
class Primary:
    bank: Bank
    number: int
    offset: float

    @property
    def solid_angle(self):
        return (self.number - 1) * self.offset + self.bank.solid_angle


def angle_to_measure(primary: Primary, max_theta: float, *, N=1024):
    th = np.linspace(0, max_theta, N)
    out = np.array(th)

    cyrstal_offset_index = np.searchsorted(th, primary.bank.offset)

    if primary.number > 1:
        if (primary.offset - primary.bank.solid_angle) < primary.bank.offset:
            raise Exception("not sure this is practical, not modeled yet")

        solid_offset_index = np.searchsorted(
            th, primary.offset - primary.bank.solid_angle
        )
        out[solid_offset_index:] += primary.solid_angle
    else:
        solid_offset_index = None
    out[cyrstal_offset_index:solid_offset_index] += primary.bank.solid_angle

    return out, th


def plot_layout(det, *, ax=None, target=100, target_time=None, frame_dt=None):
    if ax is None:
        _fig, ax = plt.subplots(layout="constrained", figsize=(4, 3))
    ax.set_aspect("equal")
    ax.set_ylim(-0.1, 1)
    ax.set_xlim(1, -0.5)
    ax.plot([0], [0], marker="o", ms=20, color="k")
    ax.axis("off")
    wedge_kwargs = {"center": (0, 0), "r": 0.9, "width": 0.5, "alpha": 0.5}
    cmap = matplotlib.colormaps["tab10"]

    min_range = min_to_theta(det, target)
    if target_time is None:
        target_time = _layout["default_target_time_s"]
    if frame_dt is None:
        frame_dt = _layout["frame_dt_s"]
    rate = min_range / target_time
    speed = 2000 * frame_dt  # deg /s

    for n, color in enumerate(cmap(np.linspace(0, 1, det.number))):
        ax.add_artist(
            Wedge(
                theta1=(det.offset * (n)),
                theta2=(det.bank.solid_angle + det.offset * (n)),
                **wedge_kwargs,
                color=color,
            )
        )

        ax.add_artist(
            Wedge(
                theta1=(det.offset * (n)) + min_range,
                theta2=(det.bank.solid_angle + det.offset * (n)) + min_range,
                center=(0, 0),
                r=0.39,
                width=0.075,
                color=color,
                alpha=0.3,
            )
        )

        base_th = det.offset * n
        for m in range(det.bank.number):
            th = np.deg2rad(base_th + m * det.bank.offset)
            ends = np.array(
                [wedge_kwargs["r"], (wedge_kwargs["r"] - 0.5 * wedge_kwargs["width"])]
            )
            ax.plot(ends * np.cos(th), ends * np.sin(th), lw=2, color="k")

    ax.annotate(
        "\n".join(
            [
                f"Number of banks: {det.number}",
                f"Bank offset: {det.offset}°",
                f"Crystal per Bank: {det.bank.number}",
                f"Crystal offset: {det.bank.offset}°",
            ]
        ),
        (0, 1),
        xycoords="axes fraction",
        xytext=(5, 15),
        textcoords="offset points",
        usetex=True,
        va="top",
    )
    ax.annotate(
        "\n".join([rf"Total Crystals \& Detectors: {det.number * det.bank.number}"]),
        (1, 1),
        xycoords="axes fraction",
        xytext=(0, 15),
        textcoords="offset points",
        usetex=True,
        va="top",
        ha="right",
    )

    ax.add_artist(
        Wedge(
            center=(0, 0),
            theta1=0,
            theta2=min_range,
            color=".5",
            r=0.39,
            width=0.075,
        )
    )

    ax.annotate(
        "\n".join(
            [
                rf"\textbf{{{min_range:.1f}°}} motion required for full pattern [{target}°]",
                rf"\textbf{{{rate:.2f}°/s}} and \textbf{{{int(np.ceil(rate / frame_dt)):d} fps}} for full pattern in {target_time}s",
                rf"Full Pattern in \textbf{{{min_range / speed:.1f}s}} [{speed}°/s]",
            ]
        ),
        (0, 0),
        xycoords="axes fraction",
        xytext=(5, -4),
        textcoords="offset points",
        va="top",
        usetex=True,
    )


def min_to_theta(det, target):
    out, th = angle_to_measure(det, 150, N=1024 * 4)
    return th[np.searchsorted(out, target)]


# %%
# Blessed primary layout: 12 crystals/bank @ 2°, 4 banks, primary spacing 29°.
layout_primary = Primary(
    Bank(_layout["crystals_per_bank"], _layout["pitch_deg"]),
    _layout["banks"],
    _layout["primary_offset_deg"],
)

fig, ax = plt.subplots(layout="constrained", figsize=(4.5, 4))
plot_layout(layout_primary, ax=ax, target=125)
_save(fig, "layout.png")

# %%

fig, ax_arr = plt.subplots(3, 2, layout="constrained", figsize=(7, 10))
for row, pitch in zip(ax_arr, [2, 2.5, 3], strict=True):
    for ax, (cry, banks) in zip(row, [(12, 4), (8, 6)], strict=True):
        layout_v = Primary(Bank(cry, pitch), banks, (cry - 1) * pitch + 5)
        plot_layout(layout_v, ax=ax, target=125)
_save(fig, "layout_grid.png")

# %%

fig, ax = plt.subplots(layout="constrained")

ax.plot(*angle_to_measure(layout_primary, 45)[::-1], color="k")
ax.set_xlim(0, 45)
ax.set_ylim(bottom=0)
ax.set_xlabel(r"arm $2\Theta$")
ax.set_ylabel(r"covered $2\Theta$")
ax.annotate(
    "'spotlight'",
    (0, 10),
    xytext=(3, 0),
    textcoords="offset points",
    va="bottom",
    ha="left",
    usetex=True,
    rotation=90,
)
ax.annotate(
    "'hi-rep'",
    (2, 24),
    arrowprops=dict(arrowstyle="-|>"),
    xytext=(0, 25),
    textcoords="offset points",
    va="bottom",
    ha="center",
    usetex=True,
)
ax.annotate(
    "'full coverage'",
    (6, 112),
    arrowprops=dict(arrowstyle="-|>"),
    xytext=(0, 15),
    textcoords="offset points",
    va="bottom",
    ha="center",
    usetex=True,
)
ax.annotate(
    "'extended'",
    (25, 100),
    xytext=(0, 0),
    textcoords="offset points",
    va="bottom",
    ha="center",
    usetex=True,
)

_save(fig, "angle_to_measure.png")
_fdr_params.maybe_show(_args)
