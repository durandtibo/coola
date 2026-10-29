r"""Common display utilities for rich."""

from __future__ import annotations

__all__ = [
    "configure_rich_logging",
    "make_progressbar",
    "make_spinner",
    "print_markdown",
    "print_pretty",
]

from typing import TYPE_CHECKING

from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.display.rich.logging import configure_rich_logging
    from coola.display.rich.print_ import print_markdown, print_pretty
    from coola.display.rich.progressbar import make_progressbar, make_spinner

# The names defined in the modules that import ``rich`` are imported the first time they are used,
# so ``rich`` is not imported with the package.
_LAZY_EXPORTS = {
    "configure_rich_logging": "coola.display.rich.logging",
    "make_progressbar": "coola.display.rich.progressbar",
    "make_spinner": "coola.display.rich.progressbar",
    "print_markdown": "coola.display.rich.print_",
    "print_pretty": "coola.display.rich.print_",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
