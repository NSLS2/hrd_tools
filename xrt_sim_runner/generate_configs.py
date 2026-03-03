import argparse
from collections import defaultdict
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, get_args, get_origin
from hashlib import md5
import uuid

import numpy as np
import tomli_w
from cycler import Cycler, cycler

from hrd_tools.config import (
    AnalyzerCalibration,
    AnalyzerConfig,
    CompleteConfig,
    DetectorConfig,
    SimConfig,
    SimScanConfig,
    SourceConfig,
    GeneratorInvocation,
)


def get_defaults(min_tth=20, max_tth=20.5):
    return CompleteConfig(
        **{
            "source": SourceConfig(
                E_incident=29_400,
                pattern_path="/home/tcaswell/scratch/2026/02/28/11bmb_7871_Y1.xye",
                dx=1,
                dz=1,
                dy=1,
                # TODO set this based on detector + R + Rd + min_tth
                delta_phi=3,
                E_hwhm=1.4e-4,
                # default to unfocused
                h_div=np.rad2deg(22e-6),
                v_div=np.rad2deg(0.9e-6),
                max_tth=max_tth,
                min_tth=min_tth,
                source_offset_x=0,
                source_offset_y=0,
                source_offset_z=0,
            ),
            "sim": SimConfig(nrays=500_000),
            "detector": DetectorConfig(
                # default to mediapix3
                pitch=0.055,
                transverse_size=256,
                height=10,
            ),
            "analyzer": AnalyzerConfig(
                R=950,
                Rd=115,
                cry_offset=np.deg2rad(2.5),
                cry_width=102,
                cry_depth=54,
                N=1,
                # will be set when buliding end station
                incident_angle=0,
                thickness=10,
                roll=0,
            ),
            "scan": SimScanConfig(start=min_tth, stop=max_tth, delta=1e-4),
        }
    )


def convert_cycler(cycle: Cycler) -> list[CompleteConfig]:
    defaults = get_defaults()
    out = []
    for entry in cycle:
        nested: dict[str, dict[str, Any]] = defaultdict(dict)
        for k, v in entry.items():
            outer, _, inner = k.partition(".")
            nested[outer][inner] = v

        out.append(
            replace(
                defaults,
                **{k: replace(getattr(defaults, k), **v) for k, v in nested.items()},
            )
        )
    return out


config_classes = [
    AnalyzerConfig,
    DetectorConfig,
    SimConfig,
    SourceConfig,
    SimScanConfig,
    AnalyzerCalibration,
]


# Helper: given a field's annotation, return its conversion function.
def get_conversion_func(annot):
    origin = get_origin(annot)
    if origin is tuple:
        args = get_args(annot)
        # Assume homogeneous tuple such as tuple[float, ...]
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        else:
            return args[0]
    return annot


# Build a mapping from configuration key to its conversion function.
type_map = {}
allowed_keys = set()
full_name_map = {}
for cfg_name_field in fields(CompleteConfig):
    cls = cfg_name_field.type
    cfg_name = cfg_name_field.name
    for field in fields(cls):
        if field.name in allowed_keys:
            raise ValueError(f"Name colision on {field.name}")
        allowed_keys.add(field.name)
        type_map[field.name] = get_conversion_func(field.type)
        full_name_map[field.name] = f"{cfg_name}.{field.name}"


# Utility: convert a comma-separated string to a list of values using conv.
def list_type(value_str: str, conv):
    items = [item.strip() for item in value_str.split(",") if item.strip()]
    try:
        return [conv(item) for item in items]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not convert values: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="Generate cycler objects from per-key command-line flags with proper type conversion.\n"
        "Flags with multiple values are combined via addition (Cartesian product), while flags "
        "with a single value are incorporated via multiplication (broadcasting the constant)."
    )
    # Add optional subdirectory flag
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Optional subdirectory name under 'configs/' to nest the generated config files.",
    )
    # Dynamically add a flag for each allowed key.
    for key in sorted(allowed_keys):
        conv = type_map[key]
        # Special case: make short_description required at argparse level
        is_required = (key == "short_description")
        parser.add_argument(
            f"--{key}",
            type=lambda s, conv=conv: list_type(s, conv),
            help=f"Comma-separated list of values for '{key}' (converted to {conv.__name__}).",
            required=is_required,
        )

    args = parser.parse_args()

    # Separate cyclers into variable (multiple values) and constant (single value).
    variable_cyclers = []
    constant_cyclers = []
    for key in sorted(allowed_keys):
        values = getattr(args, key)
        if values is not None:
            c = cycler(full_name_map[key], values)
            if len(values) == 1:
                constant_cyclers.append(c)
            else:
                variable_cyclers.append(c)

    if not variable_cyclers and not constant_cyclers:
        parser.error("At least one configuration flag must be provided.")

    # Combine variable cyclers using addition (Cartesian sum).
    combined: Cycler | None = None
    if variable_cyclers:
        combined = variable_cyclers[0]
        for c in variable_cyclers[1:]:
            combined = combined + c

    # Incorporate constant cyclers using multiplication.
    for c in constant_cyclers:
        if combined is None:
            combined = c
        else:
            combined = combined * c
    assert (
        combined is not None
    )  # Should never be None here due to earlier check, helps type checker.
    print("Generated cycler object:")
    print(combined)

    configs = convert_cycler(combined)
    config_path = Path("configs")
    if args.subdir:
        config_path = config_path / args.subdir
    config_path.mkdir(parents=True, exist_ok=True)
    for f in config_path.glob("config_*.toml"):
        f.unlink()

    gen_md = GeneratorInvocation(
        **{
            "cycler": repr(combined),
            "uuid": str(uuid.uuid4()),
            "md5sum": md5(repr(combined).encode()).hexdigest(),
        }
    )
    for j, config in enumerate(configs):
        print(
            ", ".join(
                f"{k}: {getattr(getattr(config, sub), key)}"
                for k, sub, _, key in [(k, *k.partition(".")) for k in combined.keys]
            )
        )

        with open(config_path / f"config_{j}.toml", "wb") as fout:
            tomli_w.dump({**asdict(config), "generator": asdict(gen_md)}, fout)


if __name__ == "__main__":
    main()
