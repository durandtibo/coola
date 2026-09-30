from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import pytest

from coola.utils.logging import CaptureHandler
from tests.integration.helpers import run_threads

if TYPE_CHECKING:
    from collections.abc import Iterator

LOGGER_NAME = "coola.test_capture_handler_integration"


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger(LOGGER_NAME)
    level, propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    yield logger
    logger.setLevel(level)
    logger.propagate = propagate


#####################################
#     Tests for CaptureHandler      #
#####################################


def test_capture_handler_thread_id_is_creation_thread() -> None:
    """Test that the handler is bound to the thread that created it."""
    handler = CaptureHandler()
    created: list[CaptureHandler] = []

    run_threads([threading.Thread(target=lambda: created.append(CaptureHandler()))])

    assert handler._thread_id == threading.get_ident()
    assert created[0]._thread_id != handler._thread_id


def test_capture_handler_only_captures_owner_thread(logger: logging.Logger) -> None:
    """Test that records logged from another thread are ignored."""
    handler = CaptureHandler()
    logger.addHandler(handler)
    try:
        run_threads([threading.Thread(target=lambda: logger.info("from other thread"))])
        logger.info("from owner thread")
    finally:
        logger.removeHandler(handler)

    assert [record.getMessage() for record in handler.records] == ["from owner thread"]


def test_capture_handler_concurrent_handlers_are_isolated(logger: logging.Logger) -> None:
    """Test that handlers created in different threads only capture the
    records of their own thread."""
    num_threads = 4
    num_messages = 20
    results: dict[int, list[str]] = {}
    barrier = threading.Barrier(num_threads)

    def worker(index: int) -> None:
        handler = CaptureHandler()
        logger.addHandler(handler)
        barrier.wait()
        for j in range(num_messages):
            logger.info("worker %d msg %d", index, j)
        barrier.wait()
        logger.removeHandler(handler)
        results[index] = [record.getMessage() for record in handler.records]

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(num_threads)])

    assert results == {
        i: [f"worker {i} msg {j}" for j in range(num_messages)] for i in range(num_threads)
    }
