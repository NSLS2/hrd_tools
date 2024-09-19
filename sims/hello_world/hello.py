# -*- coding: utf-8 -*-
"""

__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "2024-09-17"

Created with xrtQook




"""

import numpy as np
import sys

sys.path.append(r"/home/tcaswell/source/bnl/kklmn/xrt")
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.materials_elemental as rmatsel
import xrt.backends.raycing.materials_compounds as rmatsco
import xrt.backends.raycing.materials_crystals as rmatscr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.apertures as rapts
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing as raycing
import xrt.plotter as xrtplot
import xrt.runner as xrtrun

crystalSi01 = rmats.CrystalSi(t=1)

arm_tth = 0.2  # 0135

# copy ESRF geometry as baseline
R = 425
Rd = 370

cry_offset = np.deg2rad(2)
cry_width = 102
cry_depth = 54

E_incident = 20_000


ring_tth = 0.2
theta_b = crystalSi01.get_Bragg_angle(E_incident)


def set_crystals(arm_tth, crystals, screens, offset=cry_offset):
    print(f"{arm_tth=}")
    for j, (cry, screen) in enumerate(zip(crystals, screens)):
        cry_tth = arm_tth + j * offset
        # accept xrt coordinates
        cry_y = R * np.cos(cry_tth)
        cry_z = R * np.sin(cry_tth)
        cry.pitch = -cry_tth + theta_b
        cry.center = [0, cry_y, cry_z]

        theta_pp = np.pi / 4 - (2 * theta_b - cry_tth)
        screen.center = [
            0,
            cry_y + Rd * np.sin(theta_pp),
            cry_z - Rd * np.cos(theta_pp),
        ]


def build_beamline(N=3):
    beamLine = raycing.BeamLine(alignE=E_incident)

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        center=[0, 0, 0],
        dx=0.001,
        dz=0.001,
        distxprime=r"annulus",
        dxprime=[2499, ring_tth],
        distzprime=r"flat",
        dzprime=[np.pi / 4, 3 * np.pi / 4],
        distE="normal",
        energies=[E_incident, E_incident * 1.4e-4],
    )
    beamLine.screen_main = rscreens.Screen(
        bl=beamLine, center=[0, 150, r"auto"], name="main"
    )

    for j in range(0, N):
        cry_tth = arm_tth + j * cry_offset
        # accept xrt coordinates
        cry_y = R * np.cos(cry_tth)
        cry_z = R * np.sin(cry_tth)

        theta_pp = np.pi / 4 - (2 * theta_b - cry_tth)
        setattr(
            beamLine,
            f"oe{j:02d}",
            roes.OE(
                name=f"cry{j:02d}",
                bl=beamLine,
                center=[0, cry_y, cry_z],
                pitch=-cry_tth + theta_b,
                positionRoll=np.pi,
                material=crystalSi01,
                limPhysX=[-cry_width / 2, cry_width / 2],
                limPhysY=[-cry_depth / 2, cry_depth / 2],
            ),
        )
        setattr(
            beamLine,
            f"screen{j:02d}",
            rscreens.Screen(
                bl=beamLine,
                center=[
                    0,
                    cry_y + Rd * np.sin(theta_pp),
                    cry_z - Rd * np.cos(theta_pp),
                ],
                x=(1, 0, 0),
                # z=(...)  # TODO for tilt
            ),
        )
    # monkeypatch the number of crystals
    beamLine.N_crystals = N
    return beamLine


def run_process(beamLine):
    # "raw" beam
    geometricSource01beamGlobal01 = beamLine.geometricSource01.shine()
    screen01beamLocal01 = beamLine.screen_main.expose(
        beam=geometricSource01beamGlobal01
    )

    outDict = {
        "source": geometricSource01beamGlobal01,
        "source_screen": screen01beamLocal01,
    }

    for j in range(beamLine.N_crystals):

        oeglobal, oelocal = getattr(beamLine, f"oe{j:02d}").reflect(
            beam=geometricSource01beamGlobal01
        )
        outDict[f"cry{j:02d}_local"] = oelocal
        outDict[f"cry{j:02d}_global"] = oeglobal
        outDict[f"screen{j:02d}"] = getattr(beamLine, f"screen{j:02d}").expose(
            beam=oeglobal
        )

    # prepare flow looks at the locals of this frame so update them
    locals().update(outDict)
    beamLine.prepare_flow()

    return outDict


rrun.run_process = run_process


def gen(beamline):
    crystals = beamline.oes
    screens = beamline.screens[1:]

    start = ring_tth - (beamline.N_crystals - 1) * cry_offset
    tths = np.linspace(start, ring_tth, 512)
    set_crystals(tths[30], crystals, screens)
    # beamline.screen_main.name = f"{tth:.4f}"
    yield
    for j, tth in enumerate(tths):
        # beamline.screen_main.name = f"{tth:.4f}"
        set_crystals(tth, crystals, screens)
        beamline.glowFrameName = f"/tmp/frame_{j:04d}.png"
        yield


def show_bl():
    bl = build_beamline()
    rrun.run_process = run_process
    bl.glow(centerAt="screen01", exit_on_close=False, generator=gen, generatorArgs=[bl])
    # bl.glow(scale=[5e3, 10, 5e3], centerAt='xtal1')
    # xrtrun.run_ray_tracing(beamLine=bl, generator=gen, generatorArgs=[bl])
    return bl


def define_plots():
    plots = []
    return plots


def main():
    beamLine = build_beamline()
    E0 = list(beamLine.geometricSource01.energies)[0]
    beamLine.alignE = E0
    plots = define_plots()
    xrtrun.run_ray_tracing(plots=plots, backend=r"raycing", beamLine=beamLine)


if __name__ == "__main__":
    main()

bl = show_bl()
