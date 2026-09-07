from __future__ import annotations

import threading

from coola.identifier import ObjectIdGenerator
from tests.integration.helpers import run_threads

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
