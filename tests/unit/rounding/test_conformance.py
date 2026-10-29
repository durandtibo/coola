r"""Check every backend of ``round_values`` against the ``Decimal``
oracle, with exact comparisons."""

from __future__ import annotations

import math
from typing import Any

import pytest

from coola.rounding import round_values
from coola.testing.marks import numpy_available, polars_available, pyarrow_available
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
)
from tests.unit.rounding.helpers import HARD_VALUES, oracle_many

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

MODES = ("round", "floor", "ceil")
DECIMALS = (0, 1, 2, 5)
CASES = [(mode, decimals) for mode in MODES for decimals in DECIMALS]

# The values are wrapped in a container of the backend and unwrapped into a list of floats.
BACKENDS: dict[str, Any] = {
    "native": (lambda v: v, list),
}
if is_numpy_available():
    BACKENDS["numpy"] = (np.array, np.ndarray.tolist)
if is_torch_available():
    BACKENDS["torch"] = (lambda v: torch.tensor(v, dtype=torch.float64), torch.Tensor.tolist)
if is_pandas_available():
    BACKENDS["pandas"] = (pd.Series, pd.Series.tolist)
if is_polars_available():
    BACKENDS["polars"] = (pl.Series, pl.Series.to_list)
if is_pyarrow_available():
    BACKENDS["pyarrow"] = (pa.array, pa.Array.to_pylist)


# The cases that fail on ``main``, with the finding of ``review/rounding.md`` that explains them.
# Each step of the review removes its entries.
_ORACLE_FAILURES: dict[tuple[str, str, int], str] = {
    ("pyarrow", "round", 2): "F3",
    **{("pyarrow", "ceil", d): "F2" for d in (1, 2, 5)},
    **{("polars", m, d): "F1" for m in MODES for d in (1, 2, 5)},
}
_NUMPY_FAILURES: dict[tuple[str, str, int], str] = {
    **_ORACLE_FAILURES,
}


def _cases(failures: dict[tuple[str, str, int], str]) -> list:
    return [
        pytest.param(
            backend,
            mode,
            decimals,
            marks=(
                [pytest.mark.xfail(strict=True, reason=failures[backend, mode, decimals])]
                if (backend, mode, decimals) in failures
                else []
            ),
        )
        for backend in BACKENDS
        for mode, decimals in CASES
    ]


def _round(backend: str, values: list[float], mode: str, decimals: int) -> list[float]:
    wrap, unwrap = BACKENDS[backend]
    if backend == "native":
        return [round_values(v, mode=mode, decimals=decimals) for v in values]
    return unwrap(round_values(wrap(values), mode=mode, decimals=decimals))


def _bits(values: list[float]) -> list[tuple[float, float]]:
    # ``==`` hides the sign of a zero.
    return [(v, math.copysign(1.0, v)) for v in values]


@pytest.mark.parametrize(("backend", "mode", "decimals"), _cases(_ORACLE_FAILURES))
def test_round_values_hard_values_match_oracle(backend: str, mode: str, decimals: int) -> None:
    result = _round(backend, list(HARD_VALUES), mode, decimals)
    assert _bits(result) == _bits(oracle_many(list(HARD_VALUES), mode, decimals))


@numpy_available
@pytest.mark.parametrize(("backend", "mode", "decimals"), _cases(_NUMPY_FAILURES))
def test_round_values_backend_matches_numpy(backend: str, mode: str, decimals: int) -> None:
    expected = _round("numpy", list(HARD_VALUES), mode, decimals)
    assert _bits(_round(backend, list(HARD_VALUES), mode, decimals)) == _bits(expected)


@polars_available
@pytest.mark.xfail(strict=True, reason="F1")
def test_round_values_polars_column_matches_oracle() -> None:
    values = [-4.822, 1.2, 4.8, 0.225, -0.4]
    result = round_values(pl.Series(values), mode="round", decimals=1).to_list()
    assert result == [-4.8, 1.2, 4.8, 0.2, -0.4]


@pyarrow_available
def test_round_values_pyarrow_table_matches_oracle() -> None:
    table = pa.table({"a": [2.675, 0.125], "b": [1, 2]})
    assert round_values(table, decimals=2).to_pydict() == {"a": [2.68, 0.12], "b": [1, 2]}
