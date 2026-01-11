from unittest.mock import patch

from coola.reducer import NativeReducer, NumpyReducer, TorchReducer, auto_reducer
from coola.testing.fixtures import numpy_available, torch_available

##################################
#     Tests for auto_reducer     #
##################################


@torch_available
def test_auto_reducer_torch() -> None:
    assert isinstance(auto_reducer(), TorchReducer)


@numpy_available
def test_auto_reducer_numpy() -> None:
    with patch("coola.reducer.utils.is_torch_available", lambda: False):
        assert isinstance(auto_reducer(), NumpyReducer)


def test_auto_reducer_basic() -> None:
    with (
        patch("coola.reducer.utils.is_torch_available", lambda: False),
        patch("coola.reducer.utils.is_numpy_available", lambda: False),
    ):
        assert isinstance(auto_reducer(), NativeReducer)
