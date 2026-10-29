r"""Common display utilities for rich."""

from __future__ import annotations

__all__ = [
    "configure_rich_logging",
    "make_progressbar",
    "make_spinner",
    "print_markdown",
    "print_pretty",
]

from coola.display.rich.logging import configure_rich_logging
from coola.display.rich.print_ import print_markdown, print_pretty
from coola.display.rich.progressbar import make_progressbar, make_spinner
