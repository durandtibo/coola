from __future__ import annotations

import threading

import pytest

from coola.reducer import BaseReducer, NativeReducer, NumpyReducer, TorchReducer
from coola.testing.fixtures import numpy_available, torch_available
from tests.integration.helpers import run_python, run_threads

NUM_THREADS = 8
VALUES = [3, 1, 2, 5, 4]

REDUCERS = [
    pytest.param(NativeReducer, id="native"),
    pytest.param(NumpyReducer, id="numpy", marks=numpy_available),
    pytest.param(TorchReducer, id="torch", marks=torch_available),
]


@pytest.mark.parametrize("factory", REDUCERS)
def test_shared_reducer_returns_the_same_results_in_every_thread(
    factory: type[BaseReducer],
) -> None:
    """Test that one reducer instance shared by threads gives the same
    results for every method."""
    reducer = factory()
    results: list[tuple] = []
    lock = threading.Lock()

    def work() -> None:
        result = (
            reducer.max(VALUES),
            reducer.min(VALUES),
            reducer.mean(VALUES),
            reducer.median(VALUES),
            reducer.quantile(VALUES, [0.0, 1.0]),
            reducer.sort(VALUES),
        )
        with lock:
            results.append(result)

    run_threads([threading.Thread(target=work) for _ in range(NUM_THREADS)])
    expected = (5, 1, 3.0, 3, [1.0, 5.0], [1, 2, 3, 4, 5])
    assert results == [expected] * NUM_THREADS


def test_importing_reducer_does_not_import_numpy_or_torch() -> None:
    """Test that importing the package does not import NumPy or
    PyTorch."""
    code = (
        "import sys; import coola.reducer; "
        "print(*[m for m in ('numpy', 'torch') if m in sys.modules])"
    )
    assert run_python(code) == []
