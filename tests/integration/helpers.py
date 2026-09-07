from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading
    from collections.abc import Sequence


def run_threads(threads: Sequence[threading.Thread]) -> None:
    r"""Start every thread in ``threads`` and wait for all of them to
    finish.

    Args:
        threads: The threads to run concurrently.
    """
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
