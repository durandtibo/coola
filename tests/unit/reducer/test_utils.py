from typing import Any
from unittest.mock import Mock, patch

from coola.reducer import NativeReducer, NumpyReducer, auto_reducer
from coola.testing.marks import numpy_available

##################################
#     Tests for auto_reducer     #
##################################


@numpy_available
def test_auto_reducer_numpy() -> None:
    assert isinstance(auto_reducer(), NumpyReducer)


def test_auto_reducer_basic() -> None:
    with patch("coola.reducer.utils.is_numpy_available", lambda: False):
        assert isinstance(auto_reducer(), NativeReducer)


def _auto_reducer_with_numpy() -> tuple[Any, Mock]:
    import_object = Mock()
    with (
        patch("coola.reducer.utils.is_numpy_available", lambda: True),
        patch("coola.reducer.utils.pkgutil.resolve_name", import_object),
    ):
        reducer = auto_reducer()
    return reducer, import_object


def test_auto_reducer_returns_the_numpy_reducer_if_numpy_is_available() -> None:
    reducer, import_object = _auto_reducer_with_numpy()
    assert reducer is import_object.return_value.return_value


def test_auto_reducer_imports_the_numpy_reducer() -> None:
    _, import_object = _auto_reducer_with_numpy()
    import_object.assert_called_once_with("coola.reducer.numpy:NumpyReducer")
