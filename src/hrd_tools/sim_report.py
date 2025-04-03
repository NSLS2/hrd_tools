# %%

import base64
import io
from dataclasses import dataclass
from typing import Any

import multianalyzer
import tiled.client
from jinja2 import Template

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
