r"""Implement logging utilities."""

from __future__ import annotations

__all__ = ["CaptureHandler"]

import logging
import threading


class CaptureHandler(logging.Handler):
    r"""Collect the records logged by the thread that created the
    handler.

    Records emitted from other threads are ignored by default, so
    concurrent comparisons do not see each other's records.

    Args:
        level: The minimum level of the records to capture.
            Defaults to ``logging.INFO``.
        all_threads: If ``True``, the records of all the threads are
            captured. Defaults to ``False``.

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

    def __init__(self, level: int = logging.INFO, *, all_threads: bool = False) -> None:
        super().__init__(level=level)
        self.records: list[logging.LogRecord] = []
        self._all_threads = all_threads
        self._thread_id = threading.get_ident()

    def emit(self, record: logging.LogRecord) -> None:
        if self._all_threads or record.thread == self._thread_id:
            self.records.append(record)
