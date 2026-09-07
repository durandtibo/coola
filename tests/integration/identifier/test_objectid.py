from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from coola.identifier import ObjectIdGenerator

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_threads(threads: Sequence[threading.Thread]) -> None:
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


######################################
#     Tests for ObjectIdGenerator     #
######################################


def test_object_id_generator_generate_thread_safe() -> None:
    """Test that concurrent calls to generate() never produce duplicate
    identifiers."""
    generator = ObjectIdGenerator()
    ids: list[str] = []
    lock = threading.Lock()
    num_threads = 200

    def _worker() -> None:
        object_id = generator.generate()
        with lock:
            ids.append(object_id)

    run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(ids) == num_threads
    assert len(ids) == len(set(ids))
