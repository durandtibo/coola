from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import pytest

from coola.utils.logging import CaptureHandler

if TYPE_CHECKING:
    from collections.abc import Iterator

LOGGER_NAME = "coola.test_capture_handler"


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger(LOGGER_NAME)
    level, propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    yield logger
    logger.setLevel(level)
    logger.propagate = propagate


def make_record(level: int = logging.INFO, msg: str = "msg") -> logging.LogRecord:
    return logging.LogRecord(LOGGER_NAME, level, "", 0, msg, None, None)


#################################
#     Tests for CaptureHandler  #
#################################


def test_capture_handler_is_logging_handler() -> None:
    assert isinstance(CaptureHandler(), logging.Handler)


def test_capture_handler_level() -> None:
    assert CaptureHandler().level == logging.INFO


def test_capture_handler_records_initially_empty() -> None:
    assert CaptureHandler().records == []


def test_capture_handler_records_are_not_shared_between_instances() -> None:
    handler1, handler2 = CaptureHandler(), CaptureHandler()
    handler1.emit(make_record())
    assert len(handler1.records) == 1
    assert handler2.records == []


def test_capture_handler_emit_keeps_same_thread_records() -> None:
    handler = CaptureHandler()
    record = make_record()
    handler.emit(record)
    assert handler.records == [record]


def test_capture_handler_emit_ignores_other_threads() -> None:
    handler = CaptureHandler()
    record = make_record()
    record.thread = handler._thread_id + 1
    handler.emit(record)
    assert handler.records == []


def test_capture_handler_emit_preserves_order() -> None:
    handler = CaptureHandler()
    records = [make_record(msg=f"msg{i}") for i in range(5)]
    for record in records:
        handler.emit(record)
    assert handler.records == records


def test_capture_handler_via_logger(logger: logging.Logger) -> None:
    handler = CaptureHandler()
    logger.addHandler(handler)
    try:
        logger.info("first")
        logger.warning("second %s", "arg")
    finally:
        logger.removeHandler(handler)
    assert [record.getMessage() for record in handler.records] == ["first", "second arg"]


@pytest.mark.parametrize(
    ("level", "captured"),
    [
        (logging.DEBUG, False),
        (logging.INFO, True),
        (logging.WARNING, True),
        (logging.ERROR, True),
    ],
)
def test_capture_handler_respects_level(logger: logging.Logger, level: int, captured: bool) -> None:
    handler = CaptureHandler()
    logger.addHandler(handler)
    try:
        logger.log(level, "msg")
    finally:
        logger.removeHandler(handler)
    assert len(handler.records) == int(captured)


def test_capture_handler_keeps_extra_attributes(logger: logging.Logger) -> None:
    handler = CaptureHandler()
    logger.addHandler(handler)
    try:
        logger.info("msg", extra={"custom": (1, 2)})
    finally:
        logger.removeHandler(handler)
    assert handler.records[0].custom == (1, 2)


def test_capture_handler_thread_id_is_creation_thread() -> None:
    handler = CaptureHandler()
    assert handler._thread_id == threading.get_ident()
    created: list[CaptureHandler] = []
    thread = threading.Thread(target=lambda: created.append(CaptureHandler()))
    thread.start()
    thread.join()
    assert created[0]._thread_id != handler._thread_id


def test_capture_handler_only_captures_owner_thread(logger: logging.Logger) -> None:
    handler = CaptureHandler()
    logger.addHandler(handler)
    try:
        thread = threading.Thread(target=lambda: logger.info("from other thread"))
        thread.start()
        thread.join()
        logger.info("from owner thread")
    finally:
        logger.removeHandler(handler)
    assert [record.getMessage() for record in handler.records] == ["from owner thread"]


def test_capture_handler_concurrent_handlers_are_isolated(logger: logging.Logger) -> None:
    results: dict[int, list[str]] = {}
    barrier = threading.Barrier(4)

    def worker(i: int) -> None:
        handler = CaptureHandler()
        logger.addHandler(handler)
        barrier.wait()
        for j in range(20):
            logger.info("worker %d msg %d", i, j)
        barrier.wait()
        logger.removeHandler(handler)
        results[i] = [record.getMessage() for record in handler.records]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, messages in results.items():
        assert messages == [f"worker {i} msg {j}" for j in range(20)]


def test_capture_handler_does_not_propagate_by_itself(
    logger: logging.Logger, caplog: pytest.LogCaptureFixture
) -> None:
    handler = CaptureHandler()
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            logger.info("msg")
    finally:
        logger.removeHandler(handler)
    # The handler only records; the record still reaches other handlers.
    assert [record.getMessage() for record in handler.records] == ["msg"]
    assert caplog.messages == ["msg"]
