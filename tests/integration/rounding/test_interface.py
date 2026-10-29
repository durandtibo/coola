from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from coola.rounding import register_rounders, round_values
from coola.testing.marks import (
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    torch_available,
)
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
)
from tests.integration.helpers import run_python, run_threads

if TYPE_CHECKING:
    from pathlib import Path

if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_pyarrow_available():
    import pyarrow as pa
if is_torch_available():
    import torch


class Meters:
    def __init__(self, value: float) -> None:
        self.value = value


def round_meters(data: Meters, mode: str, decimals: int) -> Meters:
    return Meters(round_values(data.value, mode=mode, decimals=decimals))  # ty: ignore[invalid-argument-type]


########################################
#     Tests for the default registry   #
########################################


@numpy_available
def test_round_values_numpy_end_to_end() -> None:
    assert round_values(np.array([2.675, 5.5200000000000005]), decimals=2).tolist() == [2.68, 5.52]


@torch_available
def test_round_values_torch_end_to_end() -> None:
    result = round_values(torch.tensor([2.675], dtype=torch.float64), decimals=2)
    assert result.tolist() == [2.68]


@pandas_available
def test_round_values_pandas_end_to_end() -> None:
    assert round_values(pd.Series([2.675, 0.125]), decimals=2).tolist() == [2.68, 0.12]


@polars_available
def test_round_values_polars_end_to_end() -> None:
    assert round_values(pl.Series([-4.822, 12.0]), decimals=1).to_list() == [-4.8, 12.0]


@pyarrow_available
def test_round_values_pyarrow_end_to_end() -> None:
    assert round_values(pa.array([2.675, 0.125]), decimals=2).to_pylist() == [2.68, 0.12]


@torch_available
def test_round_values_torch_sparse_tensor_is_unsupported() -> None:
    sparse = torch.tensor([[1.5, 0.0]]).to_sparse()
    with pytest.raises(TypeError, match=r"Unsupported layout"):
        round_values(sparse)


@pytest.mark.usefixtures("isolated_default_registries")
def test_round_values_user_type_registered_with_register_rounders() -> None:
    register_rounders({Meters: round_meters})
    assert round_values(Meters(2.675), decimals=2).value == 2.68


@pytest.mark.usefixtures("isolated_default_registries")
def test_round_values_recursive_user_type() -> None:
    register_rounders({Meters: round_meters})
    result = round_values({"a": [Meters(1.26)]}, mode="floor", decimals=1, recursive=True)
    assert result["a"][0].value == 1.2


@dataclass
class Record:
    array: object
    series: object
    value: float


@numpy_available
@pandas_available
def test_round_values_recursive_dataclass_with_array_series_and_float() -> None:
    data = Record(np.array([1.26]), pd.Series([2.34]), 3.45)
    result = round_values(data, mode="floor", decimals=1, recursive=True)
    assert (result.array.tolist(), result.series.tolist(), result.value) == ([1.2], [2.3], 3.4)


############################################
#     Tests for the plugin group           #
############################################

PLUGIN_MODULE = """
from coola.rounding import round_values


class Marker:
    def __init__(self, value):
        self.value = value


def round_marker(data, mode, decimals):
    return Marker(round_values(data.value, mode=mode, decimals=decimals))


def register(registry):
    registry.register(Marker, round_marker)
"""


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    r"""Create an installed package (a ``dist-info`` directory) with a
    plugin of the rounders."""
    tmp_path.joinpath("coola_fake_rounder.py").write_text(PLUGIN_MODULE)
    dist_info = tmp_path.joinpath("coola_fake_rounder-1.0.dist-info")
    dist_info.mkdir()
    dist_info.joinpath("METADATA").write_text("Metadata-Version: 2.1\nName: coola-fake-rounder\n")
    dist_info.joinpath("entry_points.txt").write_text(
        "[coola.rounding.rounders]\nfake = coola_fake_rounder:register\n"
    )
    return tmp_path


def test_plugin_registers_a_rounder(plugin_dir: Path) -> None:
    """Test that the plugin of an installed package registers its
    rounder."""
    code = (
        f"import sys; sys.path.insert(0, {str(plugin_dir)!r}); "
        "from coola.rounding import round_values; "
        "from coola_fake_rounder import Marker; "
        "print(round_values(Marker(2.675), decimals=2).value)"
    )
    assert run_python(code) == ["2.68"]


########################################
#     Tests without NumPy              #
########################################


def run_without_numpy(statement: str) -> list[str]:
    r"""Run a statement in a fresh interpreter where NumPy cannot be
    imported."""
    return run_python(f"import sys; sys.modules['numpy'] = None; {statement}")


@polars_available
@pytest.mark.parametrize("data", ["pl.Series([1.26])", "pl.DataFrame({'a': [1.26]})"])
def test_round_values_polars_without_numpy_raises(data: str) -> None:
    code = (
        "import polars as pl; from coola.rounding import round_values, UnsupportedTypeError\n"
        f"data = {data}\n"
        "try:\n    round_values(data)\nexcept UnsupportedTypeError:\n    print('raised')"
    )
    assert run_without_numpy(code) == ["raised"]


@pyarrow_available
def test_round_values_pyarrow_without_numpy_raises() -> None:
    code = (
        "import pyarrow as pa; from coola.rounding import round_values, UnsupportedTypeError\n"
        "try:\n    round_values(pa.array([1.26]))\nexcept UnsupportedTypeError:\n    print('raised')"
    )
    assert run_without_numpy(code) == ["raised"]


@polars_available
def test_round_values_polars_without_numpy_ignore_returns_the_object() -> None:
    code = (
        "import polars as pl; from coola.rounding import round_values; "
        "s = pl.Series([1.26]); "
        "print(round_values(s, on_missing_package='ignore') is s)"
    )
    assert run_without_numpy(code) == ["True"]


@pyarrow_available
def test_round_values_pyarrow_without_numpy_warn_warns() -> None:
    code = (
        "import warnings, pyarrow as pa; from coola.rounding import round_values\n"
        "a = pa.array([1.26])\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    result = round_values(a, on_missing_package='warn')\n"
        "print(result is a, len(w))"
    )
    assert run_without_numpy(code) == ["True", "1"]


@polars_available
def test_round_values_recursive_without_numpy_raise_stops_the_walk() -> None:
    code = (
        "import polars as pl; from coola.rounding import round_values, UnsupportedTypeError\n"
        "try:\n    round_values([1.26, pl.Series([1.26])], recursive=True)\n"
        "except UnsupportedTypeError:\n    print('raised')"
    )
    assert run_without_numpy(code) == ["raised"]


########################################
#     Tests for the threads            #
########################################


def test_round_values_thread_safe() -> None:
    """Test that concurrent calls to round_values() on the default
    registry give the same results."""
    results: list[float] = []
    lock = threading.Lock()

    def _worker() -> None:
        value = round_values(2.675, decimals=2)
        with lock:
            results.append(value)

    run_threads([threading.Thread(target=_worker) for _ in range(50)])

    assert results == [2.68] * 50
