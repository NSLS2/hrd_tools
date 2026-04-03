# Print the ±0.1 mdeg Δ2θ tolerance bounds for all AnalyzerConfig parameters.
#
# Run with:
#   pixi run -e xrt python design_scripts/FDR/mac_sensitivity.py

from hrd_tools.sensitivity import PARAM_METADATA, create_default_config, find_parameter_bound

MAX_DELTA_TTH_MDEG = 0.1
Z_D = 15
ARM_ANGLES = [5, 88]

config = create_default_config()

print(f"Maximum parameter deviations to keep Δ2θ ≤ {MAX_DELTA_TTH_MDEG} mdeg (z_d={Z_D}mm)\n")

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
