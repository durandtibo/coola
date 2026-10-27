from coola.reducer import NativeReducer, auto_reducer
from coola.testing.fixtures import numpy_available

##################################
#     Tests for auto_reducer     #
##################################


def test_auto_reducer_mean_equals_native_reducer() -> None:
    assert auto_reducer().mean([1, 2, 3, 4]) == NativeReducer().mean([1, 2, 3, 4])


def test_auto_reducer_sort_equals_native_reducer() -> None:
    assert auto_reducer().sort([3, 1, 2]) == NativeReducer().sort([3, 1, 2])


@numpy_available
def test_auto_reducer_does_not_choose_torch_reducer() -> None:
    assert type(auto_reducer()).__name__ == "NumpyReducer"
