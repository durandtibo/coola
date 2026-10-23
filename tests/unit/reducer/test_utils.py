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


def _auto_reducer_with_torch_and_numpy() -> tuple[Mock, Mock, Mock, Mock]:
    torch_reducer = Mock(spec=TorchReducer)
    with (
        patch("coola.reducer.utils.is_torch_available", lambda: True),
        patch("coola.reducer.utils.is_numpy_available", lambda: True),
        patch("coola.reducer.utils.TorchReducer", Mock(return_value=torch_reducer)) as torch_cls,
        patch("coola.reducer.utils.NumpyReducer", Mock()) as numpy_cls,
    ):
        reducer = auto_reducer()
    return reducer, torch_reducer, torch_cls, numpy_cls


def test_auto_reducer_prioritizes_torch_over_numpy() -> None:
    reducer, torch_reducer, _, _ = _auto_reducer_with_torch_and_numpy()
    assert reducer is torch_reducer


def test_auto_reducer_creates_the_torch_reducer() -> None:
    _, _, torch_cls, _ = _auto_reducer_with_torch_and_numpy()
    torch_cls.assert_called_once_with()


def test_auto_reducer_does_not_create_the_numpy_reducer_if_torch_is_available() -> None:
    _, _, _, numpy_cls = _auto_reducer_with_torch_and_numpy()
    numpy_cls.assert_not_called()


def _auto_reducer_with_numpy_only() -> tuple[Mock, Mock, Mock]:
    numpy_reducer = Mock(spec=NumpyReducer)
    with (
        patch("coola.reducer.utils.is_torch_available", lambda: False),
        patch("coola.reducer.utils.is_numpy_available", lambda: True),
        patch("coola.reducer.utils.NumpyReducer", Mock(return_value=numpy_reducer)) as numpy_cls,
    ):
        reducer = auto_reducer()
    return reducer, numpy_reducer, numpy_cls


def test_auto_reducer_falls_back_to_numpy() -> None:
    reducer, numpy_reducer, _ = _auto_reducer_with_numpy_only()
    assert reducer is numpy_reducer


def test_auto_reducer_creates_the_numpy_reducer() -> None:
    _, _, numpy_cls = _auto_reducer_with_numpy_only()
    numpy_cls.assert_called_once_with()
