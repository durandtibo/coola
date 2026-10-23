from typing import Any
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


def _auto_reducer_with_torch_and_numpy() -> tuple[Any, Mock]:
    import_object = Mock()
    with (
        patch("coola.reducer.utils.is_torch_available", lambda: True),
        patch("coola.reducer.utils.is_numpy_available", lambda: True),
        patch("coola.reducer.utils.import_object", import_object),
    ):
        reducer = auto_reducer()
    return reducer, import_object


def test_auto_reducer_prioritizes_torch_over_numpy() -> None:
    reducer, import_object = _auto_reducer_with_torch_and_numpy()
    assert reducer is import_object.return_value.return_value


def test_auto_reducer_imports_only_the_torch_reducer_if_torch_is_available() -> None:
    _, import_object = _auto_reducer_with_torch_and_numpy()
    import_object.assert_called_once_with("coola.reducer.torch:TorchReducer")


def _auto_reducer_with_numpy_only() -> tuple[Any, Mock]:
    import_object = Mock()
    with (
        patch("coola.reducer.utils.is_torch_available", lambda: False),
        patch("coola.reducer.utils.is_numpy_available", lambda: True),
        patch("coola.reducer.utils.import_object", import_object),
    ):
        reducer = auto_reducer()
    return reducer, import_object


def test_auto_reducer_falls_back_to_numpy() -> None:
    reducer, import_object = _auto_reducer_with_numpy_only()
    assert reducer is import_object.return_value.return_value


def test_auto_reducer_imports_the_numpy_reducer() -> None:
    _, import_object = _auto_reducer_with_numpy_only()
    import_object.assert_called_once_with("coola.reducer.numpy:NumpyReducer")
