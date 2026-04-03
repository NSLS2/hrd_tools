# Sensitivity to θ_d misalignment on 2ϴ to 2θ correction
#
# This script has been superseded by the hrd_tools.sensitivity module.
# Run with:
#   pixi run -e xrt python -m hrd_tools.sensitivity -p theta_d --show
#
# To also save the figure:
#   pixi run -e xrt python -m hrd_tools.sensitivity -p theta_d --show -o theta_d_stability.png

from hrd_tools.sensitivity import main

main(["-p", "theta_d", "--show"])
