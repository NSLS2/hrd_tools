import contextlib
from dataclasses import asdict, fields, is_dataclass
import tomllib

import numpy as np
import h5py
import tqdm

from bad_tools.config import (
    AnalyzerConfig,
    DetectorConfig,
    SimConfig,
    SourceConfig,
    SimScanConfig,
)
from bad_tools.xrt.endstation import Endstation


def scan_to_file(
    bl: Endstation,
    scan_config: SimScanConfig,
    *,
    writer,
):
    cache_rate = writer.send(None)
    with contextlib.closing(writer):
        writer.send(bl)
        start, stop, delta = np.deg2rad(
            [scan_config.start, scan_config.stop, scan_config.delta]
        )
        N = int((stop - start) / delta)
        tths = np.linspace(start, stop, N)
        for batch in tqdm.tqdm(range(len(tths) // cache_rate + 1)):
            batch_tth = tths[batch * cache_rate : (batch + 1) * cache_rate]
            good = {f"screen{screen:02d}": [] for screen in range(bl.analyzer.N)}
            # bad = {f"screen{screen:02d}": [] for screen in range(bl.analyzer.N)}

            try:
                for j, tth in enumerate(batch_tth):  # noqa: B007
                    bl.set_arm(tth)
                    images, *rest = bl.get_frames()
                    for k, v in images.items():
                        good[k].append(v["good"])
                        # bad[k].append(v["bad"])
            except KeyboardInterrupt:
                if j > 0:
                    writer.send(
                        (batch_tth[:j], to_block({k: v[:j] for k, v in good.items()}))
                    )
                raise
            else:
                writer.send((batch_tth, to_block(good)))


def dump_coro(fname, cache_rate, *, tlg="sim"):
    bl = yield cache_rate
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
    while True:
        payload = yield
        with h5py.File(fname, "a") as f:
            g = f[tlg]

            if payload[0].shape[0] != payload[1].shape[0]:
                print(f"{payload[0].shape[0]} == {payload[1].shape[0]}")
            assert payload[0].shape[0] == payload[1].shape[0]
            for name, data in zip(("tth", "block"), payload, strict=True):
                ds = g[name]
                cur_len = ds.shape[0]
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
    parser.add_argument(
        "config", help="toml to read to get the configuration.", type=Path
    )
    parser.add_argument("fout", help="File to write results to.", type=Path)
    parser.add_argument(
        "--cache-rate",
        help="flush to disk every this number of angles",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    args.fout.parent.mkdir(parents=True, exist_ok=True)

    class_map = {
        "source": SourceConfig,
        "sim": SimConfig,
        "detector": DetectorConfig,
        "analyzer": AnalyzerConfig,
    }
    with args.config.open("rb") as fin:
        inp = tomllib.load(fin)

    configs = {k: cls(**inp[k]) for k, cls in class_map.items()}
    fin = Path(configs["source"].pattern_path).resolve()
    if not fin.exists():
        msg = "pattern source does not exist"
        raise ValueError(msg)

    bl = Endstation.from_configs(**configs)
    writer = dump_coro(args.fout, args.cache_rate)
    scan_config = SimScanConfig(**inp["scan"])
    scan_to_file(bl, scan_config, writer=writer)
