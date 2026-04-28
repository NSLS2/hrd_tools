"""Print maximum AnalyzerConfig parameter deviations to keep Δ2θ ≤ 0.1 mdeg.

Run with::

    pixi run -e xrt python design_scripts/FDR/mac_sensitivity.py
"""

import _fdr_params
from hrd_tools.sensitivity import (
    PARAM_METADATA,
    create_default_config,
    find_parameter_bound,
)

_args = _fdr_params.parse_args(__doc__)

MAX_DELTA_TTH_MDEG = 0.1                       # mdeg
Z_D = 15                                       # mm
ARM_ANGLES = [5, 88]                           # deg

# Use the canonical FDR energy (or CLI override) for the analyzer config.
_blessed = _fdr_params.complete_config()
_e_keV = _args.energy_keV if _args.energy_keV is not None else _blessed.source.E_incident / 1000.0
config = create_default_config(_e_keV)

print(
    f"Maximum parameter deviations to keep Δ2θ ≤ {MAX_DELTA_TTH_MDEG} mdeg (z_d={Z_D}mm)\n"
)

for param_name, meta in PARAM_METADATA.items():
    symbol = meta["unicode"]
    units = meta["units"]
    print(f"{symbol}  ({meta['description']}):")
    for arm_angle in ARM_ANGLES:
        try:
            bound = find_parameter_bound(
                config,
                param_name,
                max_delta_tth_mdeg=MAX_DELTA_TTH_MDEG,
                z_d=Z_D,
                arm_angle=arm_angle,
            )
            if units == "deg" and abs(bound) < 0.1:
                bound_str = f"{bound * 1000:.3g} mdeg"
            else:
                bound_str = f"{bound:.3g} {units}"
            print(f"  2Θ={arm_angle:2}°: {symbol} ≤ {bound_str}")
        except ValueError as e:
            print(f"  2Θ={arm_angle:2}°: {e}")
    print()
