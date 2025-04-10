# %%
import base64
import io
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.axes
import matplotlib.patches
import matplotlib.figure
import multianalyzer
import numpy as np
import tiled.client
from jinja2 import Template

from hrd_tools.config import AnalyzerConfig, DetectorConfig, SourceConfig

from .config import (
    CompleteConfig,
)
from .sim_reduction import plot_cat_fwhm_1d, plot_reduced_cat, raw_grid, reduce_catalog


@dataclass
class Reduced:
    cat: tiled.client.container.Container
    reduced: dict[tuple[str, str], tuple[multianalyzer.file_io.Result, CompleteConfig]]
    phi_max: float

    @classmethod
    def from_cat(cls, cat, phi_max):
        return cls(cat, reduce_catalog(cat, phi_max=phi_max), phi_max)


# Define the updated Jinja2 template as a multi-line string
_template_str = """
# {{ title }}

{{ description }}
## Scanned Parameters

{% if scanned_parameters %}
{% for param in scanned_parameters %}
- **{{ param.name }}:** {{ param.range }}
{% endfor %}
{% else %}
*No parameters were scanned.*
{% endif %}

## static Parameters

{% if static_parameters %}
{% for section, table in static_parameters.items() %}
### {{ section }}

| Key | Value |
| --- | ----- |{% for key, value in table.items() %}
| {{ key }} | {{ value }} |{% endfor %}

{% endfor %}
{% endif %}


## Graphs

{% if images %}
{% for image in images %}
![{{ image.caption }}]({{ image.filename }})

*{{ image.caption }}*
{% endfor %}
{% else %}
*No graphs available.*
{% endif %}
"""


def base64ify(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"


def aggregate_min_max(
    data: list[dict[str, dict[str, Any]]],
) -> dict[str, tuple[Any, Any]]:
    """
    Aggregate a list of nested dictionaries into a single dictionary.

    Each key in the output is formed by concatenating the outer and inner keys with a dot.
    The corresponding value is a tuple (min, max) representing the minimum and maximum values
    found for that inner key across all dictionaries.

    Parameters
    ----------
    data : list of dict[str, dict[str, Any]]
        A list of dictionaries where each dictionary maps an outer key (str) to an inner
        dictionary (dict[str, Any]). The inner dictionary's values should be comparable
        (e.g., numbers).

    Returns
    -------
    dict[str, tuple[Any, Any]]
        A dictionary with keys in the format "outer.inner" and values as tuples (min, max)
        computed from the aggregated inner dictionary values.

    Examples
    --------
    >>> data = [
    ...     {"A": {"x": 10, "y": 20}},
    ...     {"A": {"x": 15, "y": 25}},
    ...     {"B": {"z": 5}},
    ...     {"B": {"z": 7, "w": 3}}
    ... ]
    >>> aggregate_min_max(data)
    {'A.x': (10, 15), 'A.y': (20, 25), 'B.z': (5, 7), 'B.w': (3, 3)}
    """
    result: dict[str, tuple[Any, Any]] = {}

    for outer_dict in data:
        for outer_key, inner_dict in outer_dict.items():
            for inner_key, value in inner_dict.items():
                composite_key = f"{outer_key}.{inner_key}"
                if composite_key not in result:
                    result[composite_key] = (value, value)
                else:
                    current_min, current_max = result[composite_key]
                    result[composite_key] = (
                        min(current_min, value),
                        max(current_max, value),
                    )

    return result


def display_analyzer(
    config: AnalyzerConfig,
    fig: matplotlib.figure.SubFigure,
    *,
    equal_aspect=False,
    show_thickness=False,
):
    sample_loc = np.array([0, 0])
    theta = np.deg2rad(config.incident_angle)
    cry_loc = sample_loc + config.R * np.array([np.cos(theta), np.sin(theta)])
    det_loc = cry_loc + config.Rd * np.array([np.cos(theta), -np.sin(theta)])

    ax = fig.subplots()

    ax.plot(*np.vstack([sample_loc, cry_loc, det_loc]).T, lw=2.5, color="k")

    if show_thickness:
        rect = matplotlib.patches.Rectangle(
            cry_loc - np.array([config.cry_depth / 2, 0]),
            config.cry_depth,
            config.thickness,
            facecolor="b",
            clip_on=False,
            edgecolor="b",
            lw=10,
        )
        ax.add_artist(rect)
    else:
        ax.plot(
            [cry_loc[0] - config.cry_depth / 2, cry_loc[0] + config.cry_depth / 2],
            [cry_loc[1], cry_loc[1]],
        )

    R_text_loc = (sample_loc + cry_loc) / 2 + 2
    Rd_text_loc = (cry_loc + det_loc) / 2 - 2
    ax.text(
        *R_text_loc,
        f"R = {config.R}mm",
        rotation=config.incident_angle,
        transform_rotates_text=True,
        rotation_mode="anchor",
        ha="center",
    )
    ax.text(
        *Rd_text_loc,
        f"Rd = {config.Rd}mm",
        rotation=-config.incident_angle,
        transform_rotates_text=True,
        rotation_mode="anchor",
        ha="center",
        va="top",
    )

    ax.set_ylim(-10, cry_loc[1] + 10)
    ax.set_xlim(-10, config.R + config.Rd + 10)
    if equal_aspect:
        ax.set_aspect("equal")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    fig.suptitle("Crystal analyzer geometry")


# %%
def display_det(
    config: DetectorConfig,
    fig: matplotlib.figure.SubFigure,
):
    ax = fig.subplots()
    width = config.pitch * config.transverse_size

    rect = matplotlib.patches.Rectangle(
        (-width / 2, -config.height / 2),
        width=width,
        height=config.height,
        facecolor="none",
        edgecolor="k",
        lw=1,
    )
    ax.text(
        0,
        0,
        f"{config.transverse_size}px\n {config.pitch * 1000} μm",
        ha="center",
        va="center",
    )
    ax.annotate(
        f"{width} mm",
        (0, config.height / 2),
        xytext=(0, 5),
        textcoords="offset pixels",
        ha="center",
    )
    ax.annotate(
        f"{config.height} mm",
        (width / 2, 0),
        xytext=(5, 0),
        textcoords="offset pixels",
        ha="center",
        rotation=-90,
        rotation_mode="anchor",
    )
    ax.add_artist(rect)

    ax.set_aspect("equal")
    ax.set_xlim(-width, width)
    ax.set_ylim(-config.height, config.height)
    ax.axis("off")
    fig.suptitle("Detector Geometry")


# %%


# %%


def display_source(
    config: SourceConfig,
    fig: matplotlib.figure.SubFigure,
    div_unit: Literal["deg", "rad"] = "rad",
):
    fig_real, fig_div = fig.subfigures(2, 1, height_ratios=(1, 2))
    for sfig in (fig_real, fig_div):
        sfig.patch.set_facecolor(".9")
    fig_div.suptitle("divergence")
    fig_real.suptitle("shape")
    ax_dict = fig_div.subplot_mosaic(
        [
            ["ax_slch", "ax_slcv"],
            ["ax_recip", "ax_recip"],
        ],
    )
    ax = fig_real.subplots()
    ax.set_aspect("equal")
    if div_unit == "rad":
        unit = "μrad"
        h_div = np.deg2rad(config.h_div) * 1e6
        v_div = np.deg2rad(config.v_div) * 1e6
    elif div_unit == "deg":
        unit = "μdeg"
        h_div = config.h_div * 1e6
        v_div = config.v_div * 1e6
    else:
        raise ValueError('div_unit must be in {"deg", "rad"}')

    # real space view
    ax.annotate(
        f"depth: {config.dy} mm",
        (1, 0),
        xycoords="axes fraction",
        va="baseline",
        ha="center",
    )
    ax.xaxis.set_label_position("top")
    width = config.dx
    height = config.dz
    rect = matplotlib.patches.Rectangle(
        (-width / 2, -height / 2),
        width=width,
        height=height,
        facecolor="none",
        edgecolor="k",
        lw=1,
    )
    ax.annotate(
        f"{width} mm",
        (0, height / 2),
        xytext=(0, 5),
        textcoords="offset pixels",
        ha="center",
    )
    ax.annotate(
        f"{height} mm",
        (max(0.5, width / 2), 0),
        xytext=(5, 0),
        textcoords="offset pixels",
        # ha="center",
        # rotation=-90,
        rotation_mode="anchor",
    )
    ax.add_artist(rect)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("hoizontal")
    ax.set_ylabel("vertical")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("Source")

    # divergence heat map
    # TODO make unit dependent
    min_d, max_d = -100, 100

    def norm(bins, sigma, mu=0):
        return (
            1
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((bins - mu) ** 2) / (2 * sigma**2))
        )

    cols, rows = np.ogrid[min_d:max_d:1024j, min_d:max_d:1024j]
    if v_div > 0 and h_div > 0:
        img = norm(cols, v_div / 2.355) * norm(rows, h_div / 2.355)
    else:
        img = np.zeros((1024, 1024))
    ax_dict["ax_recip"].imshow(img, cmap="gray", extent=[min_d, max_d, min_d, max_d])

    ax_dict["ax_recip"].axis("off")

    # divergence slices

    bins = np.linspace(min_d, max_d, 1024)

    for div, ax_name, title in zip(
        [h_div, v_div], ["ax_slch", "ax_slcv"], ["h", "v"], strict=False
    ):
        sigma = div / 2.355
        assert sigma >= 0
        ax_slice = ax_dict[ax_name]
        ax_slice.set_yticks([])

        ax_slice.set_xlabel(f"angle ({unit})")
        ax_slice.set_title(f"{title}: {div:2g} {unit}")
        ax_slice.set_xlim(min_d, max_d)
        if sigma > 0:
            ax_slice.plot(bins, norm(bins, sigma), color="k")
        else:
            ax_slice.axvline(0, color="k")
            ax_slice.axhline(0, color="k")

# %%
def generate_report(r: Reduced, fname="test.md"):
    # Create a Template object from the string
    template = Template(_template_str)
    cat = r.cat
    integrations = r.reduced
    scan_md = cat.metadata["scan"]
    data = {
        "title": "Simulation Report",
        "description": cat.metadata["short_descrption"],
        "static_parameters": {
            **cat.metadata["static_values"],
            "scan": scan_md,
            "integration": {"phi_max": r.phi_max},
        },
        "scanned_parameters": [
            {"name": k, "range": f"{vmin} - {vmax}"}
            for k, (vmin, vmax) in aggregate_min_max(
                cat.metadata["varied_values"].values()
            ).items()
        ],
        "images": [
            {
                "caption": "integrations",
                "filename": (
                    base64ify(
                        plot_reduced_cat(
                            cat,
                            integrations,
                            reference_pattern=True,
                            xlim=(scan_md["start"], scan_md["stop"]),
                        )
                    )
                ),
            },
            {
                "caption": "Full Scans",
                "filename": base64ify(raw_grid(cat, integrations)),
            },
            {
                "caption": "FWHM",
                "filename": base64ify(
                    plot_cat_fwhm_1d(cat, integrations, [(7.65, 7.72), (7.975, 8.1)])[0]
                ),
            },
        ],
    }

    with open(fname, "w") as fout:
        fout.write(template.render(data))

# %%
