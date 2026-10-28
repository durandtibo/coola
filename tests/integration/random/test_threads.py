from __future__ import annotations

import threading

from coola.equality import objects_are_equal
from coola.random import get_torch_generator
from coola.testing.marks import torch_available
from coola.utils.imports import is_torch_available
from tests.integration.helpers import run_threads

if is_torch_available():
    import torch

#########################################
#     Tests for get_torch_generator     #
#########################################


@torch_available
def test_get_torch_generator_threads_produce_the_sequences_of_the_single_thread_run() -> None:
    """Test that threads that use their own generator get the numbers of
    the single-thread run."""
    seeds = [1, 2, 3, 4]
    expected = {seed: torch.rand(100, generator=get_torch_generator(seed)) for seed in seeds}
    results: dict[int, torch.Tensor] = {}
    lock = threading.Lock()

    def _worker(seed: int) -> None:
        generator = get_torch_generator(seed)
        values = torch.cat([torch.rand(10, generator=generator) for _ in range(10)])
        with lock:
            results[seed] = values

    run_threads([threading.Thread(target=_worker, args=(seed,)) for seed in seeds])

    assert objects_are_equal(results, expected)
