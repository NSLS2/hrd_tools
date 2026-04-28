from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, fields
from pathlib import Path
from typing import IO, Any, Union

import h5py
import numpy as np
import pandas as pd

from .config import (
    AnalyzerCalibration,
    AnalyzerConfig,
    CompleteConfig,
    DetectorConfig,
    SimConfig,
    SimScanConfig,
    SourceConfig,
)

# Type alias for "thing yaml.dump/load can take or be opened from"
YAMLStream = Union[str, Path, IO[str]]


def load_all_config(path: Path, *, ext="h5", prefix="") -> dict[Path, CompleteConfig]:
    configs = {}
    for _ in sorted(path.glob(f"**/{prefix}*{ext}")):
        config = load_config(_)

        configs[_] = config
    return configs


def load_config(fname, *, tlg="sim"):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        return load_config_from_group(g)


def load_config_from_group(grp):
    configs = {}

    for fld in fields(CompleteConfig):
        try:
            config_grp = grp[f"{fld.name}_config"]
        except KeyError:
            if fld.name != "scan":
                print(f"missing {fld.name}")
        else:
            config_attrs = dict(**config_grp.attrs)
            if fld.name == "analyzer" and "acceptance_angle" in config_attrs:
                config_attrs["incident_angle"] = config_attrs.pop("acceptance_angle")
            configs[fld.name] = fld.type(**config_attrs)
    if "scan" not in configs:
        tth = grp["tth"][:]
        configs["scan"] = SimScanConfig(
            start=np.rad2deg(tth[0]),
            stop=np.rad2deg(tth[-1]),
            delta=np.rad2deg(np.mean(np.diff(tth))),
        )
    return CompleteConfig(**configs)


def dflt_config(complete_config):
    print(complete_config)
    return AnalyzerCalibration(
        detector_centers=(
            [complete_config.detector.transverse_size / 2] * complete_config.analyzer.N
        ),
        psi=[
            np.rad2deg(complete_config.analyzer.cry_offset) * j
            for j in range(complete_config.analyzer.N)
        ],
        roll=[complete_config.analyzer.roll] * complete_config.analyzer.N,
    )


def load_data(fname, *, tlg="sim", scale=1):
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        block = g["block"][:]
        block *= scale
        return np.rad2deg(g["tth"][:]), block.astype("int32")


def find_varied_config(configs):
    out = defaultdict(set)
    for c in configs:
        if isinstance(c, dict):
            cd = c
        else:
            print(type(c))
            cd = asdict(c)
        for k, sc in cd.items():
            for f, v in sc.items():
                out[(k, f)].add(v)
    return [k for k, s in out.items() if len(s) > 1]


def config_to_table(config):
    df = pd.DataFrame(
        {
            k: {
                f"{outer}.{inner}": v
                for outer, cfg in asdict(v).items()
                for inner, v in cfg.items()
            }
            for k, v in config.items()
        }
    ).T.infer_objects()
    df["job"] = [str(_.name.partition("-")[0]) for _ in df.index]
    return df


# ---------------------------------------------------------------------------
# YAML I/O for CompleteConfig
# ---------------------------------------------------------------------------

# Map dataclass field name in CompleteConfig -> dataclass type.
_COMPLETE_CONFIG_FIELDS: dict[str, type] = {
    "source": SourceConfig,
    "sim": SimConfig,
    "detector": DetectorConfig,
    "analyzer": AnalyzerConfig,
    "scan": SimScanConfig,
}


def _scalar_to_yaml(v: Any) -> Any:
    """Convert dataclass-field values into YAML-friendly scalars."""
    if isinstance(v, Path):
        return str(v)
    # numpy scalars (e.g. the derived np.float64 Bragg angle) don't have a
    # PyYAML representer; cast them to plain Python scalars.
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (AttributeError, ValueError):
            pass
    return v


def _to_yaml_dict(cfg: CompleteConfig) -> dict[str, dict[str, Any]]:
    """Convert a CompleteConfig to a plain nested dict suitable for YAML dump."""
    out: dict[str, dict[str, Any]] = {}
    for fld in fields(cfg):
        sub = getattr(cfg, fld.name)
        out[fld.name] = {k: _scalar_to_yaml(v) for k, v in asdict(sub).items()}
    return out


def _coerce_field(field_type: type, name: str, value: Any) -> Any:
    """Coerce a YAML-loaded value into the type expected by the dataclass."""
    # The dataclasses currently use only float/int/str/Optional[Path], so a
    # very small set of coercions is sufficient.
    if name == "pattern_path":
        if value is None or value == "":
            return None
        return Path(value)
    # ``analyzer.incident_angle`` is the only field that is documented as
    # "null => derive at load time" in the blessed FDR YAML; we let it through
    # as NaN here so the dataclass construction doesn't fail, and the
    # ``_fdr_params`` helper replaces it with the derived value.
    if name == "incident_angle" and value is None:
        return float("nan")
    return value


def _build_subconfig(cls: type, raw: dict[str, Any]) -> Any:
    """Build a sub-dataclass from a YAML dict, dropping unknown keys."""
    valid = {f.name for f in fields(cls)}
    extra = set(raw) - valid
    if extra:
        # Auxiliary keys (not part of the dataclass) are silently ignored;
        # callers can read them off the raw YAML if they need them.
        raw = {k: v for k, v in raw.items() if k in valid}
    coerced = {k: _coerce_field(cls, k, v) for k, v in raw.items()}
    return cls(**coerced)


def complete_config_to_yaml(cfg: CompleteConfig, stream: YAMLStream) -> None:
    """Write a :class:`CompleteConfig` to YAML.

    Parameters
    ----------
    cfg : CompleteConfig
        The configuration to serialize.
    stream : str | Path | TextIO
        Output target. If a string or :class:`Path`, the file is opened (and
        closed) automatically.
    """
    import yaml

    data = _to_yaml_dict(cfg)
    with ExitStack() as stack:
        if isinstance(stream, (str, Path)):
            stream = stack.enter_context(open(stream, "w"))
        yaml.safe_dump(data, stream, sort_keys=False)


def complete_config_from_yaml(stream: YAMLStream) -> CompleteConfig:
    """Read a :class:`CompleteConfig` from YAML.

    Unknown keys in any of the top-level ``source / sim / detector /
    analyzer / scan`` blocks are silently ignored, so additional auxiliary
    blocks (e.g. ``layout``, ``crystal_reference``) can live alongside the
    blessed config in the same file.
    """
    import yaml

    with ExitStack() as stack:
        if isinstance(stream, (str, Path)):
            stream = stack.enter_context(open(stream))
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")

    parts: dict[str, Any] = {}
    for name, cls in _COMPLETE_CONFIG_FIELDS.items():
        if name not in data:
            raise ValueError(f"missing '{name}' block in YAML")
        parts[name] = _build_subconfig(cls, dict(data[name]))
    return CompleteConfig(**parts)
