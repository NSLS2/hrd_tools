# Sensitivity to θ_d misalignment on 2ϴ to 2θ correction
#
# This script has been superseded by the hrd_tools.sensitivity module.
# Run with:
#   pixi run -e xrt python -m hrd_tools.sensitivity -p theta_d --show
#
# To also save the figure:
#   pixi run -e xrt python -m hrd_tools.sensitivity -p theta_d --show -o theta_d_stability.png


import _fdr_params
from hrd_tools.sensitivity import main

_args = _fdr_params.parse_args(__doc__)

forwarded = ["-p", "theta_d"]
if _args.show:
    forwarded.append("--show")
forwarded.extend(["-o", str(_args.outdir / "theta_d_stability.png")])
if _args.energy_keV is not None:
    forwarded.extend(["-E", str(_args.energy_keV)])

_args.outdir.mkdir(parents=True, exist_ok=True)
main(forwarded)
