from __future__ import annotations

import logging

from coola.utils.logging import CaptureHandler

LOGGER_NAME = "coola.test_capture_handler"


def _record() -> logging.LogRecord:
    return logging.LogRecord(LOGGER_NAME, logging.INFO, "", 0, "msg", None, None)


def test_capture_handler_keeps_same_thread_records() -> None:
    handler = CaptureHandler()
    record = _record()
    handler.emit(record)
    assert handler.records == [record]


def test_capture_handler_ignores_other_threads() -> None:
    handler = CaptureHandler()
    record = _record()
    record.thread = handler._thread_id + 1
    handler.emit(record)
    assert handler.records == []


def test_capture_handler_level() -> None:
    assert CaptureHandler().level == logging.INFO
