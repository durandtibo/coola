from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.reducers import EmptySequenceError, ReducerRegistry, TorchReducer
from coola.testing import torch_available
from coola.utils import is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()  # pragma: no cover


if TYPE_CHECKING:
    from collections.abc import Sequence


EMPTY_SEQUENCES = [[], (), torch.tensor([])]


@torch_available
def test_reducer_registry_available_reducers() -> None:
    assert isinstance(ReducerRegistry.registry["torch"], TorchReducer)


##################################
#     Tests for TorchReducer     #
##################################


@torch_available
def test_torch_reducer_str() -> None:
    assert str(TorchReducer()).startswith("TorchReducer(")


@torch_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [2], [2, -1, -2, -3], torch.tensor([-2, -1, 0, 1, 2])],
)
def test_torch_reducer_max_int(values: Sequence[int | float]) -> None:
    val = TorchReducer().max(values)
    assert isinstance(val, int)
    assert val == 2


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [2.5],
        [2.5, -1.5, -2, -3],
        torch.tensor([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_max_float(values: Sequence[int | float]) -> None:
    val = TorchReducer().max(values)
    assert isinstance(val, float)
    assert val == 2.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_max_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the maximum because the summary is empty"
    ):
        TorchReducer().max(values)


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_mean_int(values: Sequence[int | float]) -> None:
    val = TorchReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.0


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_mean_float(values: Sequence[int | float]) -> None:
    val = TorchReducer().mean(values)
    assert isinstance(val, float)
    assert val == 0.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_mean_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the mean because the summary is empty"
    ):
        TorchReducer().mean(values)


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [0], torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_median_int(values: Sequence[int | float]) -> None:
    val = TorchReducer().median(values)
    assert isinstance(val, int)
    assert val == 0


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        [0.5],
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_median_float(values: Sequence[int | float]) -> None:
    val = TorchReducer().median(values)
    assert isinstance(val, float)
    assert val == 0.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_median_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the median because the summary is empty"
    ):
        TorchReducer().median(values)


@torch_available
@pytest.mark.parametrize(
    "values",
    [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), [-2], [-2, 1, 2, 3], torch.tensor([-2, -1, 0, 1, 2])],
)
def test_torch_reducer_min_int(values: Sequence[int | float]) -> None:
    val = TorchReducer().min(values)
    assert isinstance(val, int)
    assert val == -2


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-2.5, -1.5, 0.5, 1.5, 2.5],
        (-2.5, -1.5, 0.5, 1.5, 2.5),
        [-2.5],
        [-2.5, 1.5, 2, 3],
        torch.tensor([-2.5, -1.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_min_float(values: Sequence[int | float]) -> None:
    val = TorchReducer().min(values)
    assert isinstance(val, float)
    assert val == -2.5


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_min_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the minimum because the summary is empty"
    ):
        TorchReducer().min(values)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ],
)
def test_torch_reducer_quantile_int(values: Sequence[int | float]) -> None:
    assert TorchReducer().quantile(values, (0.2, 0.5, 0.9)) == [2, 5, 9]


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5),
        torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]),
    ],
)
def test_torch_reducer_quantile_float(values: Sequence[int | float]) -> None:
    assert TorchReducer().quantile(values, (0.0, 0.1, 0.4, 0.9)) == [0.5, 1.5, 4.5, 9.5]


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_quantile_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError, match=r"Cannot compute the quantiles because the summary is empty"
    ):
        TorchReducer().quantile(values, [0.5])


@torch_available
@pytest.mark.parametrize(
    "values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0), torch.tensor([2, 1, -2, 3, 0])]
)
def test_torch_reducer_sort_int(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values) == [-2, 0, 1, 2, 3]


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [2.5, 1.5, -2.5, 3.5, 0.5],
        (2.5, 1.5, -2.5, 3.5, 0.5),
        torch.tensor([2.5, 1.5, -2.5, 3.5, 0.5]),
    ],
)
def test_torch_reducer_sort_float(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values) == [-2.5, 0.5, 1.5, 2.5, 3.5]


@torch_available
@pytest.mark.parametrize(
    "values", [[2, 1, -2, 3, 0], (2, 1, -2, 3, 0), torch.tensor([2, 1, -2, 3, 0])]
)
def test_torch_reducer_sort_descending(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values, descending=True) == [3, 2, 1, 0, -2]


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_sort_empty(values: Sequence[int | float]) -> None:
    assert TorchReducer().sort(values) == []


@torch_available
@pytest.mark.parametrize(
    "values", [[-2, -1, 0, 1, 2], (-2, -1, 0, 1, 2), torch.tensor([-2, -1, 0, 1, 2])]
)
def test_torch_reducer_std_int(values: Sequence[int | float]) -> None:
    assert math.isclose(TorchReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@torch_available
@pytest.mark.parametrize(
    "values",
    [
        [-1.5, -0.5, 0.5, 1.5, 2.5],
        (-1.5, -0.5, 0.5, 1.5, 2.5),
        torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5]),
    ],
)
def test_torch_reducer_std_float(values: Sequence[int | float]) -> None:
    assert math.isclose(TorchReducer().std(values), 1.5811388492584229, abs_tol=1e-6)


@torch_available
@pytest.mark.parametrize("values", [[1], [1.0], torch.tensor([1]), torch.tensor([1.0])])
def test_torch_reducer_std_one(values: Sequence[int | float]) -> None:
    assert math.isnan(TorchReducer().std(values))


@torch_available
@pytest.mark.parametrize("values", EMPTY_SEQUENCES)
def test_torch_reducer_std_empty(values: Sequence[int | float]) -> None:
    with pytest.raises(
        EmptySequenceError,
        match=r"Cannot compute the standard deviation because the summary is empty",
    ):
        TorchReducer().std(values)


@torch_available
def test_torch_reducer_no_torch() -> None:
    with (
        patch("coola.utils.imports.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        TorchReducer()
