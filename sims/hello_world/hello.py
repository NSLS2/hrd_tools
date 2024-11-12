from dataclasses import dataclass, asdict


import numpy as np
import pandas as pd
import h5py


import multianalyzer.opencl


import xrt.backends.raycing as raycing
import xrt.backends.raycing.materials as rmats
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.screens as rscreens
import xrt.backends.raycing.sources as rsources
import xrt.backends.raycing.sources_beams as rsources_beams
import xrt.runner as xrtrun


@dataclass(frozen=True)
class AnalyzerConfig:
    # sample to (central) crystal
    R: float
    # crystal to detector distance
    Rd: float
    # angular offset between crystals in deg
    cry_offset: float
    # crystal width (transverse to beam) in mm
    cry_width: float
    # crystal depth (direction of beam) in mm
    cry_depth: float
    # number of crystals
    N: int
    # acceptance angle of crystals
    acceptance_angle: float


@dataclass(frozen=True)
class DetectorConfig:
    # pixel pitch in mm
    pitch: float
    # pixel width in transverse direction
    transverse_size: int
    # Size of active area in direction of beam in mm
    height: float


fname = "/home/tcaswell/Downloads/11bmb_7871_Y1.xye"


@dataclass
class SimConfig:
    nrays: int


def pattern_sample(tth, cumsum, N):
    return np.arccos(np.sin(tth)[np.searchsorted(cumsum, np.random.rand(N))])


class XrdSource(rsources.GeometricSource):
    def __init__(self, *args, pattern, **kwargs):
        super().__init__(*args, **kwargs)
        self._pattern = pattern
        self._I_cumsum = (pattern.I1 / pattern.I1.sum()).cumsum()

    def _set_annulus(self, axis1, axis2, rMin, rMax, phiMin, phiMax):
        # if rMax > rMin:
        #    A = 2. / (rMax**2 - rMin**2)
        #    r = np.sqrt(2*np.random.uniform(0, 1, self.nrays)/A + rMin**2)
        # else:
        #    r = rMax
        I_cumsum = self._I_cumsum
        # TODO trim tth/r
        tth = self._pattern.theta[
            np.searchsorted(I_cumsum, np.random.rand(self.nrays))
        ].values
        r = np.tan(np.deg2rad(tth))
        phi = np.random.uniform(phiMin, phiMax, self.nrays)
        axis1[:] = r * np.cos(phi)
        axis2[:] = r * np.sin(phi)


# because xrt.glow has a regex on the str of the type
XrdSource.__module__ = rsources.GeometricSource.__module__
crystalSi01 = rmats.CrystalSi(t=1)

arm_tth = 0.2  # 0135

# copy ESRF geometry as baseline
config_mac = AnalyzerConfig(
    R=425,
    Rd=370,
    cry_offset=np.deg2rad(2),
    cry_width=102,
    cry_depth=54,
    N=3,
    acceptance_angle=0.05651551,
)
config = AnalyzerConfig(
    R=300,
    Rd=115,
    cry_offset=np.deg2rad(3),
    cry_width=102,
    cry_depth=54,
    N=5,
    acceptance_angle=0.05651551,
)

detector_config = DetectorConfig(pitch=0.055, transverse_size=512, height=1)
sim_config = SimConfig(nrays=10_000)
E_incident = 29_400


ring_tth = np.deg2rad(15)
theta_b = crystalSi01.get_Bragg_angle(E_incident)


def set_crystals(arm_tth, crystals, screens, config):
    offset = config.cry_offset
    for j, (cry, screen) in enumerate(zip(crystals, screens, strict=True)):
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


def build_beamline(config: AnalyzerConfig, sim_config: SimConfig):
    beamLine = raycing.BeamLine(alignE=E_incident)

    reference_pattern = pd.read_csv(
        fname,
        skiprows=3,
        names=["theta", "I1", "I0"],
        sep=" ",
        skipinitialspace=True,
        index_col=False,
    )
    delta_phi = np.pi / 8
    beamLine.geometricSource01 = XrdSource(
        bl=beamLine,
        center=[0, 0, 0],
        dx=1,
        dz=0.1,
        distxprime=r"annulus",
        dxprime=[ring_tth - 5e-3, ring_tth + 5e-3],
        distzprime=r"flat",
        dzprime=[np.pi / 2 - delta_phi, np.pi / 2 + delta_phi],
        distE="normal",
        energies=[E_incident, E_incident * 1.4e-4],
        pattern=reference_pattern,
        nrays=sim_config.nrays,
    )
    # TODO switch to plates
    beamLine.screen_main = rscreens.Screen(
        bl=beamLine, center=[0, 150, r"auto"], name="main"
    )

    for j in range(config.N):
        cry_tth = arm_tth + j * config.cry_offset
        # accept xrt coordinates
        cry_y = config.R * np.cos(cry_tth)
        cry_z = config.R * np.sin(cry_tth)

        pitch = (-cry_tth + theta_b,)
        theta_pp = np.pi / 4 - (2 * theta_b - cry_tth)

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
                targetOpenCL=(0, 0),
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

    main = rsources_beams.Beam(copyFrom=geometricSource01beamGlobal01)

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
        # main = rsources_beams.Beam(copyFrom=geometricSource01beamGlobal01)
        # main.concatenate(oeglobal)
        main = oeglobal
        outDict[f"screen_soure{j:02d}"] = main
        outDict[f"screen{j:02d}"] = getattr(beamLine, f"screen{j:02d}").expose(
            beam=main
        )
        del main

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


def show_bl(config: AnalyzerConfig, sim_config: SimConfig):
    bl = build_beamline(config, sim_config)
    rrun.run_process = run_process
    bl.glow(centerAt="screen01", exit_on_close=False, generator=gen, generatorArgs=[bl])
    # bl.glow(scale=[5e3, 10, 5e3], centerAt='xtal1')
    # xrtrun.run_ray_tracing(beamLine=bl, generator=gen, generatorArgs=[bl])
    return bl


bl = show_bl(config, sim_config)


def build_hist(lb, *, isScreen=True, pixel_size=0.055, shape=(448, 512)):
    # print(lb.x, lb.y, lb.z, lb.state)
    good = (lb.state == 1) | (lb.state == 2)
    if isScreen:
        x, y = lb.x[good], lb.z[good]
    else:
        x, y = lb.x[good], lb.y[good]

    limits = list((pixel_size * np.array([[-0.5, 0.5]]).T * np.array([shape])).T)

    flux = lb.Jss[good] + lb.Jpp[good]
    hist2d, yedges, xedges = np.histogram2d(
        y, x, bins=shape, range=limits, weights=flux
    )

    return hist2d, yedges, xedges


def scan(
    bl: raycing.BeamLine,
    start: float,
    stop: float,
    delta: float,
    detector: DetectorConfig,
):
    import tqdm

    tths = np.linspace(start, stop, int((stop - start) / delta))
    out = {f"screen{screen:02d}": [] for screen in range(bl.config.N)}
    for tth in tqdm.tqdm(tths):
        move_arm(bl, tth)
        outDict = run_process(bl)
        for screen in out:
            lb = outDict[screen]
            # TODO thread detector config through
            out[screen].append(
                build_hist(
                    lb,
                    pixel_size=detector.pitch,
                    shape=(
                        int(detector.height // detector.pitch),
                        detector.transverse_size,
                    ),
                )[0].sum(axis=0)
            )

    return {k: np.asarray(v) for k, v in out.items()}, tths


def show2(data, tth, *, N=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    tth = np.rad2deg(tth)

    if N is not None:
        data = np.random.poisson(N * (data / np.max(data)))
        kwargs = dict(vmin=1)
        label = "count"
    else:
        kwargs = dict(norm="log")
        label = "I [arb]"

    sub = data
    fig = plt.figure(figsize=(9, 4.5), constrained_layout=True)

    (ax2, ax1) = fig.subplots(1, 2, sharey=True, width_ratios=[1, 5])
    ax2.set_ylabel("arm position (°)")
    ax2.set_xlabel("row sum")
    ax1.set_xlabel("detector column")

    ax2.plot(sub.sum(axis=1), tth)
    # cmap = plt.get_cmap('viridis').resampled(N)
    # cmap.set_under('w', alpha=0)
    # norm = BoundaryNorm(np.arange(.5, N_photons + 1, 1), N_photons)
    # im = ax1.imshow(sub, extent=(0, sub.shape[-1], tth[start], tth[stop]), aspect='auto', norm=norm, cmap=cmap, origin='lower')#, interpolation_stage='rgba')

    im = ax1.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[0, data.shape[1], tth[0], tth[-1]],
        interpolation_stage="rgba",
        cmap=mpl.colormaps["viridis"].with_extremes(under="w"),
        **kwargs,
    )
    cb = fig.colorbar(im, use_gridspec=True, extend="min", label=label)
    # cb.ax.set_yticks(np.arange(1, N_photons + 1, 1))
    cb.ax.yaxis.set_ticks([], minor=True)


def show(data, tths, *, N=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(layout="constrained")
    if N is not None:
        data = np.random.poisson(N * (data / np.max(data)))
        kwargs = dict(vmin=1)
        label = "count"
    else:
        kwargs = dict(norm="log")
        label = "I [arb]"

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=[0, data.shape[1], tths[0], tths[-1]],
        interpolation_stage="rgba",
        cmap=mpl.colormaps["viridis"].with_extremes(under="w"),
        **kwargs,
    )
    cbar = fig.colorbar(im)
    cbar.set_label(label)
    ax.set_xlabel("pixel")
    ax.set_ylabel(r"arm 2$\theta$ [rad]")


def scan_and_plot(bl, tths):
    data, _ = scan(bl, tths)
    show(data, tths)


def demo():
    a, tths = scan(bl, ring_tth - 100e-4, ring_tth + 75e-4, 5e-5)
    show(a, tths)


def source_size_scan(bl):
    out = []
    widths = np.logspace(-2, 1, 16, base=10)
    for w in widths:
        bl.sources[0].dx = w
        a, tths = scan(bl, ring_tth - 100e-4, ring_tth + 75e-4, 5e-5)
        out.append(a)
    return widths, tths, np.array(out)


def dump(fname, data, tth, config, E, *, tlg="sim"):
    with h5py.File(fname, "x") as f:
        g = f.create_group(tlg)
        analyzer_config = g.create_group("analyzer_config")
        analyzer_config.attrs.update(asdict(config))
        g.attrs["E"] = E

        g["tth"] = tth
        for k, v in data.items():
            g.create_dataset(k, data=v, chunks=True, shuffle=True)


def load(fname, *, tlg="sim"):
    data = {}
    with h5py.File(fname, "r") as f:
        g = f[tlg]
        for d in g:
            if not d.startswith("screen"):
                continue
            data[d] = g[d][:]
        E = g.attrs["E"]
        config = AnalyzerConfig(**dict(g["analyzer_config"].attrs.items()))
        tths = g["tth"][:]

    return data, tths, config, E


def to_photons(data, tths, N, analyzer, detector):
    block = np.zeros((len(tths), analyzer.N, 1, detector.transverse_size))
    for j in range(analyzer.N):
        block[:, j, :, :] = data[f"screen{j:02d}"][:, None, :]

    return np.random.poisson(N * (block / np.max(block)))


def reduce_raw(
    block,
    tths,
    tth_min,
    tth_max,
    dtth,
    analyzer: AnalyzerConfig,
    detector: DetectorConfig,
    # calibration: AnalyzerCalibration,
):
    multianalyzer.opencl.OclMultiAnalyzer.NUM_CRYSTAL = np.int32(analyzer.N)
    mma = multianalyzer.opencl.OclMultiAnalyzer(
        # sample to crystal
        L=analyzer.R,
        # crystal to detector
        L2=analyzer.Rd,
        # pixel size
        pixel=detector.pitch,
        # pixel column direct beam is on
        # TODO pull from calibration structure
        center=[detector.transverse_size / 2] * analyzer.N,
        # acceptance angle of crystals
        tha=analyzer.acceptance_angle,
        # global offset of MCA vs arm position
        # TODO pull from calibration structure
        thd=0.0,
        # offsets of each crystal from origin of MCA
        # TODO pull from calibration structure
        psi=np.rad2deg([analyzer.cry_offset * j for j in range(analyzer.N)]),
        # mis-orientation of the analyzer along x (°)
        # TODO pull from calibration structure
        rollx=[0.0] * analyzer.N,
        # TODO pull from calibration structure
        # mis-orientation of the analyzer along y (°)
        rolly=[0.0] * analyzer.N,
        device=["0"],
    )
    return mma.integrate(
        # data all stacked as one
        roicollection=block,
        # arm positions
        arm=np.rad2deg(tths),
        # IO, currently constant
        mon=np.ones_like(tths),
        # range and resolution for binning
        tth_min=tth_min,
        tth_max=tth_max,
        dtth=dtth,
        # shape of "ROIs"
        num_row=detector.transverse_size,
        num_col=1,
        # how to understand the shape of roicollection
        columnorder=1,
    )


def bench_mark(res, df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(df.theta, df.I1 / df.I1.max(), label="source", zorder=15, lw=3, ls="--")
    # for j, (I, n) in enumerate(zip(res.signal, res.norm)):
    #     ax.plot(res.tth, I/I.max(), label=f'cry {j}', alpha=.5)

    val = res.signal.squeeze().sum(axis=0) / res.norm.sum(axis=0)
    print(val.shape, df.theta.shape)
    ax.plot(res.tth, val / np.nanmax(val), label="average", zorder=5, lw=1)

    ax.legend()
