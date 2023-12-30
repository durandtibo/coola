import pytest

from coola import Reduction
from coola.reducers import BaseReducer, BasicReducer, TorchReducer
from coola.testing import torch_available

###############################
#     Tests for Reduction     #
###############################


def test_reduction_reducer() -> None:
    assert isinstance(Reduction.reducer, BaseReducer)


@torch_available
def test_reduction_reducer_torch() -> None:
    assert isinstance(Reduction.reducer, TorchReducer)


def test_available_reducers() -> None:
    assert "basic" in Reduction.available_reducers()


def test_check_reducer() -> None:
    Reduction.check_reducer("basic")


def test_check_reducer_missing() -> None:
    with pytest.raises(RuntimeError, match="Incorrect reducer"):
        Reduction.check_reducer("missing")


def test_initialize_basic() -> None:
    Reduction.initialize("basic")
    assert isinstance(Reduction.reducer, BasicReducer)


@torch_available
def test_initialize_torch() -> None:
    Reduction.initialize("torch")
    assert isinstance(Reduction.reducer, TorchReducer)
