from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from coola.reducer import (
    BaseReducer,
    EmptySequenceError,
    NativeReducer,
    NumpyReducer,
    TorchReducer,
)
from coola.testing.fixtures import numpy_available, torch_available

if TYPE_CHECKING:
    from collections.abc import Sequence

REDUCERS: list[BaseReducer] = [NativeReducer(), NumpyReducer(), TorchReducer()]


@numpy_available
@torch_available
@pytest.mark.parametrize("reducer", REDUCERS)
@pytest.mark.parametrize("values", [[1], [1.0], (1,), (1.0,)])
def test_reducers_std_one_returns_nan(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must agree that the standard
    deviation of a single-element sequence is ``nan``, not an
    exception."""
    assert math.isnan(reducer.std(values))


@numpy_available
@torch_available
@pytest.mark.parametrize("reducer", REDUCERS)
@pytest.mark.parametrize("values", [[], ()])
def test_reducers_std_empty_raises(reducer: BaseReducer, values: Sequence[int | float]) -> None:
    r"""All ``BaseReducer`` implementations must raise
    ``EmptySequenceError`` for an empty sequence."""
    with pytest.raises(EmptySequenceError, match=r"the sequence is empty"):
        reducer.std(values)


@numpy_available
@torch_available
@pytest.mark.parametrize("reducer", REDUCERS)
def test_reducers_std_multiple_values_consistent(reducer: BaseReducer) -> None:
    r"""All ``BaseReducer`` implementations must agree on the standard
    deviation for a sequence with more than one value."""
    assert math.isclose(reducer.std([-2, -1, 0, 1, 2]), 1.5811388300841898, abs_tol=1e-6)
