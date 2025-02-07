import collections
import os
import warnings
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, Self

import h5py
import numpy
from tiled.adapters.array import ArrayAdapter
from tiled.adapters.resource_cache import with_resource_cache
from tiled.adapters.utils import IndexersMixin
from tiled.catalog.orm import Node
from tiled.iterviews import ItemsView, KeysView, ValuesView
from tiled.structures.array import ArrayStructure
from tiled.structures.core import Spec, StructureFamily
from tiled.structures.data_source import DataSource
from tiled.type_aliases import JSON
from tiled.utils import node_repr, path_from_uri

from bad_tools.file_io import load_config_from_group

SWMR_DEFAULT = bool(int(os.getenv("TILED_HDF5_SWMR_DEFAULT", "0")))
INLINED_DEPTH = int(os.getenv("TILED_HDF5_INLINED_CONTENTS_MAX_DEPTH", "7"))


class SimAdapter(collections.abc.Mapping[str, ArrayAdapter], IndexersMixin):
    """
    Read an HDF5 file or a group within one.

    This map the structure of an HDF5 file onto a "Tree" of array structures.

    Examples
    --------

    From the root node of a file given a filepath

    >>> import h5py
    >>> HDF5Adapter.from_uri("file://localhost/path/to/file.h5")

    From the root node of a file given an h5py.File object

    >>> import h5py
    >>> file = h5py.File("path/to/file.h5")
    >>> HDF5Adapter.from_file(file)

    From a group within a file

    >>> import h5py
    >>> file = h5py.File("path/to/file.h5")
    >>> HDF5Adapter(file["some_group']["some_sub_group"])

    """

    structure_family = StructureFamily.container

    def __init__(
        self,
        file: Any,
        *,
        structure: ArrayStructure | None = None,  # noqa: ARG002
        metadata: JSON | None = None,
        specs: list[Spec] | None = None,
    ) -> None:
        self._group = file["sim"]
        self.specs = specs or []
        self._metadata = metadata or {}

    @classmethod
    def from_catalog(
        cls,
        data_source: DataSource,
        node: Node,
        /,
        swmr: bool = SWMR_DEFAULT,
        libver: str = "latest",
        **kwargs: Any | None,
    ) -> "SimAdapter":
        assets = data_source.assets
        if len(assets) == 1:
            data_uri = assets[0].data_uri
        else:
            for ast in assets:
                if ast.parameter == "data_uri":
                    data_uri = ast.data_uri
                    break
        filepath = path_from_uri(data_uri)
        cache_key = (h5py.File, filepath, "r", swmr, libver)
        file = with_resource_cache(
            cache_key, h5py.File, filepath, "r", swmr=swmr, libver=libver
        )

        adapter = cls(
            file,
            structure=data_source.structure,
            metadata=node.metadata_,
            specs=node.specs,
        )
        dataset = kwargs.get("dataset") or kwargs.get("path") or []
        for segment in dataset:
            adapter = adapter.get(segment)
            if adapter is None:
                raise KeyError(segment)

        return adapter

    @classmethod
    def from_uris(
        cls,
        data_uri: str,
        *,
        swmr: bool = SWMR_DEFAULT,
        libver: str = "latest",
    ) -> "SimAdapter":
        filepath = path_from_uri(data_uri)
        cache_key = (h5py.File, filepath, "r", swmr, libver)
        file = with_resource_cache(
            cache_key, h5py.File, filepath, "r", swmr=swmr, libver=libver
        )
        return cls(file)

    def __repr__(self) -> str:
        return node_repr(self, list(self))

    def structure(self) -> None:
        return None

    def metadata(self) -> JSON:
        return {**self._metadata, **asdict(load_config_from_group(self._group))}

    def __iter__(self) -> Iterator[Any]:
        yield from self._group

    def __getitem__(self, key: str) -> ArrayAdapter:
        value = self._group[key]

        md = dict(value.attrs)
        if value.dtype == numpy.dtype("O"):
            warnings.warn(
                f"The dataset {key} is of object type, using a "
                "Python-only feature of h5py that is not supported by "
                "HDF5 in general. Read more about that feature at "
                "https://docs.h5py.org/en/stable/special.html. "
                "Consider using a fixed-length field instead. "
                "Tiled will serve an empty placeholder, unless the "
                "object is of size 1, where it will attempt to repackage "
                "the data into a numpy array.",
                stacklevel=2,
            )

            check_str_dtype = h5py.check_string_dtype(value.dtype)
            if check_str_dtype.length is None:
                dataset_names = value.file[self._group.name + "/" + key][...][()]
                if value.size == 1:
                    arr = numpy.array(dataset_names)
                    return ArrayAdapter.from_array(arr, metadata=md)
            return ArrayAdapter.from_array(numpy.array([]), metadata=md)
        return ArrayAdapter.from_array(value, metadata=md)

    def __len__(self) -> int:
        return 2

    def keys(self) -> KeysView:
        return KeysView(lambda: len(self), self._keys_slice)

    def values(self) -> ValuesView:
        return ValuesView(lambda: len(self), self._items_slice)

    def items(self) -> ItemsView:
        return ItemsView(lambda: len(self), self._items_slice)

    def search(self, query: Any) -> None:
        """

        Parameters
        ----------
        query :

        Returns
        -------
                Return a Tree with a subset of the mapping.

        """
        raise NotImplementedError

    def read(self, fields: str | None = None) -> Self:
        """

        Parameters
        ----------
        fields :

        Returns
        -------

        """
        if fields is not None:
            raise NotImplementedError
        return self

    # The following two methods are used by keys(), values(), items().

    def _keys_slice(self, start: int, stop: int, direction: int) -> list[Any]:
        """

        Parameters
        ----------
        start :
        stop :
        direction :

        Returns
        -------

        """
        keys = ["tth", "block"]
        if direction < 0:
            keys = list(reversed(keys))
        return keys[start:stop]

    def _items_slice(
        self, start: int, stop: int, direction: int
    ) -> list[tuple[Any, Any]]:
        """

        Parameters
        ----------
        start :
        stop :
        direction :

        Returns
        -------

        """
        items = [(key, self[key]) for key in list(self)]
        if direction < 0:
            items = list(reversed(items))
        return items[start:stop]

    def inlined_contents_enabled(self, depth: int) -> bool:
        return depth <= INLINED_DEPTH


def read_sim_output(
    filepath,
    metadata=None,  # noqa: ARG001
    **kwargs,
):
    return SimAdapter.from_uris(filepath)
