from unittest.mock import Mock, patch

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


def test_auto_reducer_prefers_torch_when_torch_and_numpy_available() -> None:
    torch_reducer = Mock(spec=TorchReducer)
    with (
        patch("coola.reducer.utils.is_torch_available", lambda: True),
        patch("coola.reducer.utils.is_numpy_available", lambda: True),
        patch("coola.reducer.utils.TorchReducer", Mock(return_value=torch_reducer)) as torch_cls,
        patch("coola.reducer.utils.NumpyReducer", Mock()) as numpy_cls,
    ):
        assert auto_reducer() is torch_reducer
        torch_cls.assert_called_once_with()
        numpy_cls.assert_not_called()


def test_auto_reducer_uses_numpy_when_torch_unavailable_and_numpy_available() -> None:
    numpy_reducer = Mock(spec=NumpyReducer)
    with (
        patch("coola.reducer.utils.is_torch_available", lambda: False),
        patch("coola.reducer.utils.is_numpy_available", lambda: True),
        patch("coola.reducer.utils.NumpyReducer", Mock(return_value=numpy_reducer)) as numpy_cls,
    ):
        assert auto_reducer() is numpy_reducer
        numpy_cls.assert_called_once_with()
