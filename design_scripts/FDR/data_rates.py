import subprocess
from collections import namedtuple
from pathlib import Path

from multihead.file_io import open_data

record = namedtuple(
    "record", "name,num_frames,density,disk_size_sparse,disk_size_dense"
)

root = Path("/home/tcaswell/data/hrd/aps/Oct25/raw/v3")

res = []

for f in root.glob("*"):
    print(f)
    try:
        raw = open_data(f, 3)

        disk_size = int(
            subprocess.run(["du", "-s", f], stdout=subprocess.PIPE, check=True)
            .stdout.decode()
            .split("\t")[0]
        )
        disk_size_dense = sum(
            int(_.split("\t")[0])
            for _ in subprocess.run(
                ["du", "-s", *(f.parent.parent / "v1").glob(f"{f.name}*")],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .split("\n")
            if _
        )
        if raw._sparse_data.density > 1e-4:
            res.append(
                record(
                    f.name,
                    raw._sparse_data.shape[1],
                    raw._sparse_data.density,
                    disk_size,
                    disk_size_dense,
                )
            )
        print(res[-1])
    except FileNotFoundError:
        ...

per_frame_dense = []
per_frame_spares = []

for rec in res:
    print(
        f"{rec.disk_size_dense / (rec.num_frames * 12):.2f} kb/frame dense\t"
        f"{rec.disk_size_sparse / (rec.num_frames * 12):.2f}  kb/frame sparse\t"
        f"{100 * rec.density:.2f}% density"
    )
