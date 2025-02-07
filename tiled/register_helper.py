import pathlib
import sys
from collections import defaultdict

from custom import SimAdapter
from tiled.client import from_uri
from tiled.client.register import dict_or_none, ensure_uri
from tiled.structures.core import StructureFamily
from tiled.structures.data_source import Asset, DataSource, Management

from bad_tools.file_io import find_varied_config

if __name__ == "__main__":
    base_path = pathlib.Path(sys.argv[1])
    client = from_uri("http://localhost:8000", api_key="secret")["raw"]
    # nuke it for now to be safe
    client.delete_tree()

    for elm in base_path.iterdir():
        if elm.is_dir():
            tiled_objs_by_run = defaultdict(list)
            for output_file in elm.glob("*h5"):
                f_uri = ensure_uri(output_file)
                adapter = SimAdapter.from_uris(f_uri)
                ds = DataSource(
                    structure_family=adapter.structure_family,
                    mimetype="application/x-hdf5",
                    structure=dict_or_none(adapter.structure()),
                    parameters={},
                    management=Management.external,
                    assets=[
                        Asset(
                            data_uri=f_uri,
                            is_directory=False,
                            parameter="data_uri",
                        )
                    ],
                )
                run, _, oid = output_file.stem.partition("-")
                tiled_objs_by_run[run].append((adapter, ds, oid))
            for run, results in tiled_objs_by_run.items():
                results.sort(key=lambda x: int(x[-1]))
                all_config = [r[0].metadata() for r in results]
                varied = find_varied_config(all_config)
                static = {}

                for k, sc in all_config[0].items():
                    static[k] = {}
                    for f, v in sc.items():
                        if (k, f) in varied:
                            continue
                        static[k][f] = v
                varied_values = {oid: {} for *_, oid in results}
                for k, f in varied:
                    for (*_, oid), config in zip(results, all_config, strict=True):
                        varied_values[oid].setdefault(k, {})
                        varied_values[oid][k][f] = config[k][f]

                client.new(
                    key=run,
                    structure_family=StructureFamily.container,
                    metadata={
                        "varied_key": [".".join(_) for _ in varied],
                        "static_values": static,
                        "varied_values": varied_values,
                    },
                    data_sources=[],
                )
                run_node = client[run]
                for res in results:
                    adapter, ds, key = res
                    run_node.new(
                        structure_family=adapter.structure_family,
                        data_sources=[ds],
                        metadata=adapter.metadata(),
                        key=key,
                    )


# if False:
#     # POST /api/v1/register/{path}
#     client.new(
#         structure_family=StructureFamily.array,
#         data_sources=[
#             DataSource(
#                 management=Management.external,
#                 mimetype="multipart/related;type=image/tiff",
#                 structure_family=StructureFamily.array,
#                 structure=structure,
#                 assets=[
#                     Asset(
#                         data_uri="file:///path/to/image1.tiff",
#                         is_directory=False,
#                         parameter="data_uri",
#                         num=1,
#                     ),
#                     Asset(
#                         data_uri="file:///path/to/image2.tiff",
#                         is_directory=False,
#                         parameter="data_uri",
#                         num=2,
#                     ),
#                 ],
#             ),
#         ],
#         metadata={},
#         specs=[],
#     )
