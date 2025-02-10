import importlib.metadata

import hrd_tools as m


def test_version():
    assert importlib.metadata.version("hrd_tools") == m.__version__
