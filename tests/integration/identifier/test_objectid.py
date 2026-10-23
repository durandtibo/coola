from __future__ import annotations

import threading
import time
from unittest.mock import patch

from coola.identifier import ObjectIdGenerator, objectid as module
from coola.utils.singleton import LazySingleton
from tests.integration.helpers import run_threads

#######################################
#     Tests for ObjectIdGenerator     #
#######################################


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


###########################################
#     Tests for get_default_generator     #
###########################################


def test_get_default_generator_thread_safe() -> None:
    """Test that concurrent first calls build a single generator."""
    built: list[ObjectIdGenerator] = []

    def _factory() -> ObjectIdGenerator:
        time.sleep(0.01)
        generator = module.ObjectIdGenerator()
        built.append(generator)
        return generator

    generators: list[ObjectIdGenerator] = []
    lock = threading.Lock()
    num_threads = 8

    def _worker() -> None:
        generator = module.get_default_generator()
        with lock:
            generators.append(generator)

    with patch.object(module, "_default_generator", LazySingleton(_factory)):
        run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(built) == 1
    assert len(generators) == num_threads
    assert all(generator is built[0] for generator in generators)
