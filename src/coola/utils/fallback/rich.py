r"""Contain fallback implementations used when ``rich`` dependency is
not available."""

from __future__ import annotations

__all__ = [
    "MINIMAL",
    "ROUNDED",
    "BarColumn",
    "Markdown",
    "MofNCompleteColumn",
    "Panel",
    "Pretty",
    "Progress",
    "RichHandler",
    "SpinnerColumn",
    "TaskProgressColumn",
    "TextColumn",
    "TimeElapsedColumn",
    "TimeRemainingColumn",
    "get_console",
]

from typing import Any

from coola.utils.fallback.factory import make_fake_class, make_fake_function

FakeClass: type = make_fake_class("rich")

BarColumn = FakeClass
Markdown = FakeClass
MofNCompleteColumn = FakeClass
Panel = FakeClass
Pretty = FakeClass
Progress = FakeClass
RichHandler = FakeClass
SpinnerColumn = FakeClass
TaskProgressColumn = FakeClass
TextColumn = FakeClass
TimeElapsedColumn = FakeClass
TimeRemainingColumn = FakeClass
get_console = make_fake_function("rich")

# The boxes are not called, they are passed to ``Panel``, which raises the error.
MINIMAL: Any = None
ROUNDED: Any = None
