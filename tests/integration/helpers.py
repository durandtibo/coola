from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import coola

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


def run_python(code: str) -> list[str]:
    r"""Run Python code in a fresh interpreter and return the words it
    prints.

    A fresh interpreter is needed to test what an import loads: the
    modules already imported by this process would hide it.

    Args:
        code: The code to run, e.g. ``"import sys; print('torch' in sys.modules)"``.

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
        env={**os.environ, "PYTHONPATH": src},
    )
    return result.stdout.split()
