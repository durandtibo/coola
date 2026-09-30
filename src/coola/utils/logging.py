r"""Implement logging utilities."""

from __future__ import annotations

__all__ = ["CaptureHandler"]

import logging
import threading


class CaptureHandler(logging.Handler):
    r"""Collect the records logged by the current thread.

    Records emitted from other threads are ignored, so concurrent
    comparisons do not see each other's records.

    Attributes:
        records: The captured records, in emission order.

    Example:
        ```pycon
        >>> import logging
        >>> from coola.utils.logging import CaptureHandler
        >>> handler = CaptureHandler()
        >>> logger = logging.getLogger("capture_example")
        >>> logger.addHandler(handler)
        >>> logger.warning("hello")
        >>> [record.getMessage() for record in handler.records]
        ['hello']
        >>> logger.removeHandler(handler)

        ```
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []
        self._thread_id = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread_id:
            self.records.append(record)
