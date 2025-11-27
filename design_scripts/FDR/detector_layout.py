# %% [markdown]
# # Detecor layout
#
# This generates a schematic

# %%
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge


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

    # from offset between the crystals to the solid angle of the bank, only have to move the offset
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


# plot2()
def plot_layout(det, *, ax=None, target=100):
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
    target_time = 30
    A = 1e-4
    rate = min_range / target_time
    speed = 2000 * A  # deg /s

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
    # ax.add_artist(w)
    # w = Wedge(
    #     center=(0, 0),
    #         theta1=min_range,
    #         theta2=min_range + det.bank.solid_angle,
    #         color='.2',
    #         r=wedge_kwargs['r'] - wedge_kwargs['width'] - .1,
    #         width=.1,
    #     )
    # ax.add_artist(w)

    ax.annotate(
        "\n".join(
            [
                rf"\textbf{{{min_range:.1f}°}} motion required for full pattern [{target}°]",
                rf"\textbf{{{rate:.2f}°/s}} and \textbf{{{int(np.ceil(rate / A)):d} fps}} for full pattern in {target_time}s",
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
layout = Primary(Bank(12, 2.5), 4, 11 * 2.5 + 5)
# layout = Primary(Bank(8, 2), 6, 7*2 + 5)
# layout = Primary(Bank(48, 2.5), 1, 0)

fig, ax = plt.subplots(layout="constrained", figsize=(4.5, 4))
plot_layout(layout, ax=ax, target=125)
fig.savefig("/tmp/layout.png", dpi=300)
plt.show()

# %%

fig, ax_arr = plt.subplots(3, 2, layout="constrained", figsize=(7, 10))
for row, pitch in zip(ax_arr, [2, 2.5, 3], strict=True):
    for ax, (cry, banks) in zip(row, [(12, 4), (8, 6)], strict=True):
        layout = Primary(Bank(cry, pitch), banks, (cry - 1) * pitch + 5)
        plot_layout(layout, ax=ax, target=125)
fig.savefig("/tmp/layout_grid.png", dpi=300)
plt.show()
