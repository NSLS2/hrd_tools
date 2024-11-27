from dataclasses import asdict, fields, is_dataclass


import numpy as np
import h5py
import tqdm

from bad_tools.config import AnalyzerConfig, DetectorConfig, SimConfig, SourceConfig
from bad_tools.xrt.endstation import Endstation


def scan_to_file(
    bl: Endstation,
    start: float,
    stop: float,
    delta: float,
    *,
    writer,
):
    cache_rate = writer.send(None)
    writer.send(bl)
    start, stop, delta = np.deg2rad([start, stop, delta])
    N = int((stop - start) / delta)
    tths = np.linspace(start, stop, N)
    for batch in tqdm.tqdm(range(len(tths) // cache_rate + 1)):
        batch_tth = tths[batch * cache_rate : (batch + 1) * cache_rate]
        good = {f"screen{screen:02d}": [] for screen in range(bl.analyzer.N)}
        # bad = {f"screen{screen:02d}": [] for screen in range(bl.analyzer.N)}

        for tth in tqdm.tqdm(batch_tth):
            bl.set_arm(tth)
            images, *rest = bl.get_frames()
            for k, v in images.items():
                good[k].append(v["good"])
                # bad[k].append(v["bad"])
        writer.send((batch_tth, to_block(good)))
    writer.close()


def dump_coro(fname, cache_rate, *, tlg="sim"):
    print("a")
    bl = yield cache_rate
    print(f"{bl=}")
    with h5py.File(fname, "x") as f:
        g = f.create_group(tlg)
        for fld in fields(bl):
            if is_dataclass(fld.type):
                cfg_g = g.create_group(f"{fld.name}_config")
                cfg_g.attrs.update(asdict(getattr(bl, fld.name)))
        g.create_dataset(
            "block",
            chunks=(cache_rate, bl.analyzer.N, 1, bl.detector.transverse_size),
            maxshape=(None, bl.analyzer.N, 1, bl.detector.transverse_size),
            shape=(0, bl.analyzer.N, 1, bl.detector.transverse_size),
            shuffle=True,
            compression="gzip",
        )
        g.create_dataset(
            "tth",
            chunks=(cache_rate,),
            maxshape=(None,),
            shape=(0,),
            shuffle=True,
            compression="gzip",
        )
    print(f"{fname=}")
    while True:
        payload = yield
        with h5py.File(fname, "a") as f:
            g = f[tlg]
            for name, data in zip(("tth", "block"), payload, strict=True):
                ds = g[name]
                cur_len = ds.shape[0]
                print(f"{data=}")
                ds.resize(cur_len + data.shape[0], axis=0)
                ds[cur_len:] = data


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


def to_block(data, *, sum_col=True):
    block = np.stack([v for k, v in sorted(data.items())], axis=1)
    if sum_col:
        block = block.sum(axis=2, keepdims=True)
    return block


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run some sims.")
    parser.add_argument("fout", help="File to write results to.", type=Path)
    parser.add_argument("start", help="start angle in deg", type=float)
    parser.add_argument("stop", help="stop angle in deg", type=float)
    parser.add_argument("delta", help="angle steps in deg", type=float)
    parser.add_argument(
        "--cache-rate",
        help="flush to disk every this number of angles",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    args.fout.parent.mkdir(parents=True, exist_ok=True)
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

    bl = Endstation.from_configs(config, source_config, detector_config, sim_config)
    writer = dump_coro(args.fout, args.cache_rate)
    scan_to_file(bl, args.start, args.stop, args.delta, writer=writer)
