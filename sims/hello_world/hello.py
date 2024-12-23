from dataclasses import asdict, fields, is_dataclass

import h5py
import multianalyzer.opencl
import numpy as np
import xrt.backends.raycing.run as rrun
import xrt.backends.raycing.sources_beams as rsources_beams

from bad_tools.config import AnalyzerConfig, DetectorConfig, SimConfig, SourceConfig
from bad_tools.xrt.endstation import Endstation

# copy ESRF geometry as baseline
config_mac = AnalyzerConfig(
    R=425,
    Rd=370,
    cry_offset=np.deg2rad(2),
    cry_width=102,
    cry_depth=54,
    N=3,
    acceptance_angle=0.05651551,
    thickness=1,
)
config_sirius = AnalyzerConfig(
    R=425,
    Rd=370,
    cry_offset=np.deg2rad(2),
    cry_width=30,  # transverse
    cry_depth=50,
    N=8,
    acceptance_angle=0.05651551,
    thickness=1,
)

config = AnalyzerConfig(
    R=300,
    Rd=115,
    cry_offset=np.deg2rad(2.5),
    cry_width=102,
    cry_depth=54,
    N=3,
    acceptance_angle=0.05651551,
    thickness=1,
)
detector_config = DetectorConfig(pitch=0.055, transverse_size=512, height=1)
sim_config = SimConfig(nrays=1_000_000)

source_config = SourceConfig(
    E_incident=29_400,
    pattern_path="/home/tcaswell/Downloads/11bmb_7871_Y1.xye",
    dx=1,
    dz=0.1,
    dy=0,
    delta_phi=np.pi / 8,
    E_hwhm=1.4e-4,
)


def show_bl(bl: Endstation):
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
        N = len([oe for oe in beamLine.oes if oe.name.startswith("cry")])

        for j in range(N):
            oeglobal, oelocal = getattr(beamLine, f"oe{j:02d}").reflect(
                beam=geometricSource01beamGlobal01
            )
            outDict[f"cry{j:02d}_local"] = oelocal
            outDict[f"cry{j:02d}_global"] = oeglobal

            outDict[f"baffle{j:0d}_local"] = getattr(
                beamLine, f"baffle{j:02d}"
            ).propagate(beam=oeglobal)

            outDict[f"screen{j:02d}"] = getattr(beamLine, f"screen{j:02d}").expose(
                beam=oeglobal
            )

        # prepare flow looks at the locals of this frame so update them
        locals().update(outDict)
        beamLine.prepare_flow()

        return outDict

    def gen(beamline: Endstation):
        ring_tth = np.deg2rad(10)
        start = ring_tth - (beamline.analyzer.N - 1) * beamline.analyzer.cry_offset
        tths = np.linspace(start, ring_tth, 128)
        beamline.set_arm(tths[30])
        # beamline.screen_main.name = f"{tth:.4f}"
        yield
        for j, tth in enumerate(tths):
            # beamline.screen_main.name = f"{tth:.4f}"
            beamline.set_arm(tth)
            beamline.bl.glowFrameName = f"/tmp/frame_{j:04d}.png"
            yield

    rrun.run_process = run_process
    bl.bl.glow(
        centerAt="screen01", exit_on_close=False, generator=gen, generatorArgs=[bl]
    )
    # bl.glow(scale=[5e3, 10, 5e3], centerAt='xtal1')
    # xrtrun.run_ray_tracing(beamLine=bl, generator=gen, generatorArgs=[bl])


bl = Endstation.from_configs(config, source_config, detector_config, sim_config)
# show_bl(bl)


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
    bl: Endstation,
    start: float,
    stop: float,
    delta: float,
):
    import tqdm

    start, stop, delta = np.deg2rad([start, stop, delta])
    tths = np.linspace(start, stop, int((stop - start) / delta))
    good = {f"screen{screen:02d}": [] for screen in range(bl.analyzer.N)}
    bad = {f"screen{screen:02d}": [] for screen in range(bl.analyzer.N)}
    for tth in tqdm.tqdm(tths):
        bl.set_arm(tth)
        images, *rest = bl.get_frames()
        for k, v in images.items():
            good[k].append(v["good"])
            bad[k].append(v["bad"])

    return (
        {k: np.asarray(v) for k, v in good.items()},
        {k: np.asarray(v) for k, v in bad.items()},
        tths,
    )


def show2(data, tth, *, N=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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
    import matplotlib as mpl
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


def source_size_scan(bl, ring_tth):
    out = []
    widths = np.logspace(-2, 1, 16, base=10)
    for w in widths:
        bl.sources[0].dx = w
        a, tths = scan(bl, ring_tth - 100e-4, ring_tth + 75e-4, 5e-5)
        out.append(a)
    return widths, tths, np.array(out)


def dump(fname, data, tth, bl, *, tlg="sim"):
    with h5py.File(fname, "x") as f:
        g = f.create_group(tlg)
        for fld in fields(bl):
            if is_dataclass(fld.type):
                cfg_g = g.create_group(f"{fld.name}_config")
                cfg_g.attrs.update(asdict(getattr(bl, fld.name)))

        g["tth"] = tth

        g.create_dataset(
            "block", data=to_block(data), chunks=True, shuffle=True, compression="gzip"
        )


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


def to_block(data, *, sum_col=True):
    block = np.stack([v for k, v in sorted(data.items())], axis=1)
    if sum_col:
        block = block.sum(axis=2, keepdims=True)
    return block


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
    phi_max: float = 90,
    # calibration: AnalyzerCalibration,
):
    cls = multianalyzer.MultiAnalyzer
    # cls = multianalyzer.opencl.OclMultiAnalyzer
    # cls.NUM_CRYSTAL = np.int32(analyzer.N)
    mma = cls(
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
        phi_max=phi_max,
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


def overlay(screen):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.imshow(screen["bad"], aspect="auto", cmap="gray_r")
    ax.imshow(screen["good"], aspect="auto", alpha=0.5)


# res = scan(bl, 7.6, 8.2, 1e-3, detector_config)
