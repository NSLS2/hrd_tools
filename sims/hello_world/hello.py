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

from bad_tools.reduce import AnalyzerConfig
crystalSi01 = rmats.CrystalSi(t=1)

arm_tth = 0.2  # 0135

config = AnalyzerConfig(
    # copy ESRF geometry as baseline
    R=425,
    Rd=370,
    cry_offset=2,
    cry_width=102,
    cry_depth=54,
    N=3,
    acceptance_angle=0.05651551
)

E_incident = 29_400


ring_tth = np.deg2rad(15)
theta_b = crystalSi01.get_Bragg_angle(E_incident)


def set_crystals(arm_tth, crystals, screens, config):
    offset = np.deg2rad(config.cry_offset)
    print(f"{arm_tth=}")
    for j, (cry, screen) in enumerate(zip(crystals, screens)):
        cry_tth = arm_tth + j * offset
        # accept xrt coordinates
        cry_y = config.R * np.cos(cry_tth)
        cry_z = config.R * np.sin(cry_tth)
        pitch = -cry_tth + theta_b

        cry.center = [0, cry_y, cry_z]
        cry.pitch = pitch

        theta_pp = theta_b + pitch
        screen.center = [
            0,
            cry_y + config.Rd * np.cos(theta_pp),
            cry_z - config.Rd * np.sin(theta_pp),
        ]

        screen_angle = theta_pp

        screen.z = (0, np.sin(screen_angle), np.cos(screen_angle))


def build_beamline(config):
    beamLine = raycing.BeamLine(alignE=E_incident)

    beamLine.geometricSource01 = rsources.GeometricSource(
        bl=beamLine,
        center=[0, 0, 0],
        dx=0.001,
        dz=0.001,
        distxprime=r"annulus",
        dxprime=[ring_tth - 5e-3, ring_tth - 5e-3],
        distzprime=r"flat",
        dzprime=[np.pi / 4, 3 * np.pi / 4],
        distE="normal",
        energies=[E_incident, E_incident * 1.4e-4],
    )
    # TODO switch to plates
    beamLine.screen_main = rscreens.Screen(
        bl=beamLine, center=[0, 150, r"auto"], name="main"
    )

    for j in range(0, config.N):
        cry_tth = arm_tth + j * config.cry_offset
        # accept xrt coordinates
        cry_y = config.R * np.cos(cry_tth)
        cry_z = config.R * np.sin(cry_tth)

        pitch = (-cry_tth + theta_b,)
        theta_pp = np.pi / 4 - (2 * theta_b - cry_tth)

        print(f"{pitch=}")
        setattr(
            beamLine,
            f"oe{j:02d}",
            roes.OE(
                name=f"cry{j:02d}",
                bl=beamLine,
                center=[0, cry_y, cry_z],
                pitch=pitch,
                positionRoll=np.pi,
                material=crystalSi01,
                limPhysX=[-config.cry_width / 2, config.cry_width / 2],
                limPhysY=[-config.cry_depth / 2, config.cry_depth / 2],
            ),
        )
        setattr(
            beamLine,
            f"screen{j:02d}",
            rscreens.Screen(
                bl=beamLine,
                center=[
                    0,
                    cry_y + config.Rd * np.cos(theta_pp),
                    cry_z - 0 * config.Rd * np.sin(theta_pp),
                ],
                x=(1, 0, 0),
            ),
        )
    # monkeypatch the config object
    beamLine.config = config
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

    for j in range(beamLine.config.N):

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


def move_arm(beamline, tth):
    set_crystals(tth, beamline.oes, beamline.screens[1:], beamline.config)


def gen(beamline):
    start = ring_tth - (beamline.config.N - 1) * beamline.config.cry_offset
    tths = np.linspace(start, ring_tth, 128)
    move_arm(beamline, tths[30])
    # beamline.screen_main.name = f"{tth:.4f}"
    yield
    for j, tth in enumerate(tths):
        # beamline.screen_main.name = f"{tth:.4f}"
        move_arm(beamline, tth)
        beamline.glowFrameName = f"/tmp/frame_{j:04d}.png"
        yield


def show_bl(config):
    bl = build_beamline(config)
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

bl = show_bl(config)


def build_hist(lb, *, isScreen=True, pixel_size=0.055, shape=(448, 512)):
    # print(lb.x, lb.y, lb.z, lb.state)
    good = (lb.state == 1) | (lb.state == 2)
    if isScreen:
        x, y, z = lb.x[good], lb.z[good], lb.y[good]
    else:
        x, y, z = lb.x[good], lb.y[good], lb.z[good]
    goodlen = len(lb.x[good])

    limits = list((pixel_size * np.array([[-0.5, 0.5]]).T * np.array([shape])).T)

    flux = lb.Jss[good] + lb.Jpp[good]
    hist2d, yedges, xedges = np.histogram2d(
        y, x, bins=shape, range=limits, weights=flux
    )

    return hist2d, yedges, xedges


def scan(bl, start, stop, delta, *, screen=0):
    tths = np.linspace(start, stop, int((stop - start) / delta))
    out = []
    for tth in tths:
        move_arm(bl, tth)
        outDict = run_process(bl)
        lb = outDict[f"screen{screen:02d}"]
        out.append(build_hist(lb)[0].sum(axis=0))

    return np.asarray(out), tths


def show(data, tths):
    fig, ax = plt.subplots(layout="constrained")
    im = ax.imshow(
        data,
        norm="log",
        aspect="auto",
        origin="lower",
        extent=[0, data.shape[1], tths[0], tths[-1]],
        interpolation_stage="rgba",
    )
    cbar = fig.colorbar(im)
    cbar.set_label("I [arb]")
    ax.set_xlabel("pixel")
    ax.set_ylabel(r"arm 2$\theta$ [rad]")


def scan_and_plot(bl, tths):
    data, _ = scan(bl, tths)
    show(data, tths)


def demo():
    a, tths = scan(bl, ring_tth - 100e-4, ring_tth + 75e-4, 5e-5)
    show(a, tths)
