"""Sensitivity to χ misalignment on 2Θ → 2θ correction.

This is a thin wrapper that defers to ``hrd_tools.sensitivity``.

Run with::

    pixi run -e xrt python design_scripts/FDR/chi_sensativity.py [--outdir ...] [--no-show] [--dpi 300]
"""

import _fdr_params
from hrd_tools.sensitivity import main

_args = _fdr_params.parse_args(__doc__)

forwarded = ["-p", "crystal_roll"]
if _args.show:
    forwarded.append("--show")
forwarded.extend(["-o", str(_args.outdir / "chi_stability.png")])
if _args.energy_keV is not None:
    forwarded.extend(["-E", str(_args.energy_keV)])

_args.outdir.mkdir(parents=True, exist_ok=True)
main(forwarded)
