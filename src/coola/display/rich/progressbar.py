r"""Provide helpers for creating progress bars with rich."""

from __future__ import annotations

__all__ = ["make_progressbar", "make_spinner"]

from typing import TYPE_CHECKING

from coola.utils.imports import is_rich_available

if TYPE_CHECKING or is_rich_available():  # pragma: no cover
    from rich import get_console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
else:  # pragma: no cover
    from coola.utils.fallback.rich import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        get_console,
    )


def make_progressbar(*, transient: bool = False) -> Progress:
    """Create a standardised Rich progress bar for use across the
    codebase.

    Builds a `Progress` instance with a consistent column layout:
    - A spinner indicating the task is active.
    - Description text.
    - A bar showing percentage complete.
    - The percentage as a number (e.g. '42%').
    - An M-of-N counter (e.g. '42/100').
    - Elapsed time since the task started.
    - Estimated time remaining.

    Using this factory ensures all progress bars in the codebase share the
    same appearance. Use as a context manager to start and stop rendering.

    Args:
        transient: If True, the progress bar is cleared from the terminal
            when the context manager exits. Useful for short-lived
            tasks where the bar would clutter the output. Defaults
            to False.

    Returns:
        A configured `Progress` instance, ready to use as a context manager.

    Raises:
        MissingPackageError: If ``rich`` is not installed.

    Example:
        ```pycon
        >>> from coola.display.rich import make_progressbar
        >>> rows = list(range(10))
        >>> with make_progressbar() as progress:
        ...     task = progress.add_task("Processing papers", total=len(rows))
        ...     for row in rows:
        ...         print(row)
        ...         progress.advance(task)
        ...

        ```
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=transient,
    )


def make_spinner(*, transient: bool = True) -> Progress:
    """Create a Rich spinner for tasks where the total number of items
    is unknown.

    Builds a :class:`~rich.progress.Progress` instance with a column
    layout suited for indeterminate tasks:

    - A spinner indicating the task is active.
    - Description text.
    - An M-of-N counter showing items processed so far (e.g. ``42/?``).
    - Processing speed (e.g. ``12.3/s``).
    - Elapsed time since the task started.

    Args:
        transient: If ``True``, the spinner is cleared from the terminal
            when the context manager exits. Useful for short-lived tasks
            where the spinner would clutter the output. Defaults to
            ``True``.

    Returns:
        A configured :class:`~rich.progress.Progress` instance ready to
        use as a context manager.

    Raises:
        MissingPackageError: If ``rich`` is not installed.

    Example:
        ```pycon
        >>> from coola.display.rich import make_spinner
        >>> items = list(range(10))
        >>> with make_spinner() as progress:
        ...     task = progress.add_task("Processing items...", total=None)
        ...     for item in items:
        ...         progress.advance(task)
        ...

        ```
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=get_console(),
        transient=transient,
    )
