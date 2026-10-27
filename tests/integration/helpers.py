from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import coola

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable, Sequence


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


def run_python(code: str, env: dict[str, str] | None = None) -> list[str]:
    r"""Run Python code in a fresh interpreter and return the words it
    prints.

    A fresh interpreter is needed to test what an import loads: the
    modules already imported by this process would hide it.

    Args:
        code: The code to run, e.g. ``"import sys; print('torch' in sys.modules)"``.
        env: Extra environment variables of the interpreter. Defaults to
            ``None``.

    Returns:
        The words printed on the standard output.

    Raises:
        subprocess.CalledProcessError: If the code fails.
    """
    src = str(Path(coola.__file__).parents[1])
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": src, **(env or {})},
    )
    return result.stdout.split()


def run_in_children(num_children: int, function: Callable[[], str]) -> list[str]:
    """Run ``function`` in ``num_children`` forked children and return
    the string each one returns."""
    results = []
    for _ in range(num_children):
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover  (measured in the child process only)
            os.close(read_fd)
            # A child that blocks (e.g. on a lock) is killed, and returns an empty string.
            signal.alarm(5)
            try:
                os.write(write_fd, function().encode())
            finally:
                os._exit(0)
        os.close(write_fd)
        with os.fdopen(read_fd) as file:
            results.append(file.read())
        os.waitpid(pid, 0)
    return results
