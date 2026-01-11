import pytest

from coola import Reduction
from coola.reducer import BaseReducer, NativeReducer, TorchReducer
from coola.testing.fixtures import torch_available

###############################
#     Tests for Reduction     #
###############################


def test_reduction_reducer() -> None:
    assert isinstance(Reduction.reducer, BaseReducer)


@torch_available
def test_reduction_reducer_torch() -> None:
    assert isinstance(Reduction.reducer, TorchReducer)


def test_available_reducers() -> None:
    assert "native" in Reduction.available_reducers()


def test_check_reducer() -> None:
    Reduction.check_reducer("native")


def test_check_reducer_missing() -> None:
    with pytest.raises(RuntimeError, match=r"Incorrect reducer"):
        Reduction.check_reducer("missing")


def test_initialize_basic() -> None:
    Reduction.initialize("native")
    assert isinstance(Reduction.reducer, NativeReducer)


@torch_available
def test_initialize_torch() -> None:
    Reduction.initialize("torch")
    assert isinstance(Reduction.reducer, TorchReducer)
