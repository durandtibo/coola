"""Provide helpers to print an object or a Markdown string with rich."""

from __future__ import annotations

__all__ = ["print_markdown", "print_pretty"]

from typing import TYPE_CHECKING, Any

from coola.utils.imports import is_rich_available

if TYPE_CHECKING:
    from rich.align import AlignMethod
    from rich.console import Console

from coola.validation import validate_ge

if TYPE_CHECKING or is_rich_available():  # pragma: no cover
    from rich import get_console
    from rich.box import MINIMAL, ROUNDED
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.pretty import Pretty
else:  # pragma: no cover
    from coola.utils.fallback.rich import (
        MINIMAL,
        ROUNDED,
        Markdown,
        Panel,
        Pretty,
        get_console,
    )


_MIN_KEPT_RATIO = 0.5
_FENCES = ("```", "~~~")


def _close_code_fence(text: str) -> str:
    r"""Close the code fence that is left open at the end of ``text``.

    Args:
        text: The Markdown text, possibly cut inside a fenced code block.

    Returns:
        ``text`` unchanged if every code fence is closed, otherwise
        ``text`` followed by a line with the closing fence.

    Example:
        ```pycon
        >>> from coola.display.rich.print_ import _close_code_fence
        >>> _close_code_fence("~~~\nx = 1")
        '~~~\nx = 1\n~~~'
        >>> _close_code_fence("~~~\nx = 1\n~~~")
        '~~~\nx = 1\n~~~'

        ```
    """
    opened = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if opened is None:
            opened = next((f for f in _FENCES if stripped.startswith(f)), None)
        elif stripped.startswith(opened):
            opened = None
    return text if opened is None else f"{text}\n{opened}"


def _truncate_markdown(msg: str, max_length: int) -> str:
    r"""Truncate a Markdown string and append a notice with the number of
    characters that are not shown.

    The text is cut at the last paragraph break (``\n\n``) of the first
    ``max_length`` characters, else at the last line break, else at the
    last space, each one only if it keeps at least half of
    ``max_length`` characters. Otherwise it is cut at ``max_length``.
    A code fence left open by the cut is closed. The cut can still leave
    an inline marker (bold, italic or code span) unbalanced when it falls
    inside a paragraph.

    Args:
        msg: The Markdown string to truncate.
        max_length: The maximum number of characters of ``msg`` to
            keep. It must be greater than or equal to 1.

    Returns:
        ``msg`` unchanged if it has at most ``max_length`` characters,
        otherwise the kept text followed by the notice
        ``*… (N more characters)*`` as a separate paragraph.

    Example:
        ```pycon
        >>> from coola.display.rich.print_ import _truncate_markdown
        >>> print(_truncate_markdown("one\n\ntwo\n\nthree", 10))
        one
        <BLANKLINE>
        two
        <BLANKLINE>
        *… (7 more characters)*

        ```
    """
    if len(msg) <= max_length:
        return msg
    kept = msg[:max_length]
    for separator in ("\n\n", "\n", " "):
        index = kept.rfind(separator)
        if index >= max_length * _MIN_KEPT_RATIO:
            kept = kept[:index]
            break
    kept = kept.rstrip()
    omitted = len(msg) - len(kept)
    return f"{_close_code_fence(kept)}\n\n*… ({omitted:,} more characters)*"


def _print_in_panel(
    renderable: Any,
    *,
    title: str | None,
    title_align: AlignMethod,
    box: bool,
    panel: bool,
    console: Console | None,
) -> None:
    r"""Render a Rich renderable to the console, optionally inside a
    panel.

    Shared implementation for :func:`print_markdown` and
    :func:`print_pretty`, factored out to keep their panel-construction
    logic (border style, title alignment, console fallback) in one
    place.

    Args:
        renderable: The Rich renderable to print (e.g. a
            :class:`~rich.markdown.Markdown` or
            :class:`~rich.pretty.Pretty` instance).
        title: Optional title displayed on the panel border.  Pass
            ``None`` to render a panel with no title.  Ignored if
            ``panel`` is ``False``.
        title_align: Horizontal alignment of ``title`` along the panel
            border (``"left"``, ``"center"``, or ``"right"``).  Ignored
            if ``panel`` is ``False``.
        box: If ``True``, render the panel with a visible rounded
            border.  If ``False``, render with a minimal
            (near-invisible) border.  Ignored if ``panel`` is ``False``.
        panel: If ``True``, wrap ``renderable`` in a
            :class:`~rich.panel.Panel`.  If ``False``, print
            ``renderable`` directly and ignore ``title``, ``title_align``,
            and ``box``.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Pass ``None`` to use the shared instance.
    """
    console = console or get_console()
    if not panel:
        console.print(renderable)
        return
    console.print(
        Panel(renderable, title=title, title_align=title_align, box=ROUNDED if box else MINIMAL)
    )


def print_markdown(
    msg: str,
    *,
    title: str | None = None,
    title_align: AlignMethod = "left",
    box: bool = True,
    panel: bool = True,
    max_length: int | None = None,
    console: Console | None = None,
) -> None:
    r"""Render a Markdown string to the console, optionally inside a Rich
    panel.

    Prints ``msg`` as rendered Markdown, wrapped in a
    :class:`~rich.panel.Panel` unless ``panel`` is ``False``.  Uses the
    provided ``console`` if given, otherwise falls back to the shared
    instance.

    Args:
        msg: The Markdown content to render.  May contain any Markdown
            syntax supported by :class:`~rich.markdown.Markdown`.
            An empty string renders a blank panel.
        title: Optional title displayed on the panel border.  Pass
            ``None`` (the default) to render a panel with no title.
            Ignored if ``panel`` is ``False``.
        title_align: Horizontal alignment of ``title`` along the panel
            border (``"left"``, ``"center"``, or ``"right"``).  Defaults
            to ``"left"``.  Ignored if ``panel`` is ``False``.
        box: If ``True`` (the default), render the panel with a visible
            rounded border.  If ``False``, render with a minimal
            (near-invisible) border.  Ignored if ``panel`` is ``False``.
        panel: If ``True`` (the default), wrap the rendered Markdown in
            a :class:`~rich.panel.Panel`.  If ``False``, print it
            directly and ignore ``title``, ``title_align``, and ``box``.
        max_length: Maximum number of characters of ``msg`` to render
            before truncating. The cut prefers a paragraph break, then a
            line break, then a space, closes a code fence left open, and
            appends a notice with the number of omitted characters. An
            inline marker (``**``) can be left unbalanced when the cut
            falls inside a paragraph. Pass ``None`` (the default) to
            render ``msg`` in full.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Overrides the shared default for this
            invocation only.  Pass ``None`` (the default) to use the
            shared instance.

    Raises:
        MissingPackageError: If ``rich`` is not installed.

    Raises:
        MissingPackageError: If ``rich`` is not installed.
        ValueError: If ``max_length`` is lower than 1.

    Example:
        ```pycon
        >>> from io import StringIO
        >>> from rich.console import Console
        >>> from coola.display.rich import print_markdown
        >>> console = Console(file=StringIO(), width=20)
        >>> print_markdown("**hello**", panel=False, console=console)
        >>> console.file.getvalue().strip()
        'hello'
        >>> print_markdown("**hello**", title="Demo")
        >>> print_markdown("**hello**", title="Demo", box=False)
        >>> print_markdown("**hello**", panel=False)

        ```
    """
    if max_length is not None:
        validate_ge(max_length, 1, name="max_length")
        msg = _truncate_markdown(msg, max_length)
    _print_in_panel(
        Markdown(msg), title=title, title_align=title_align, box=box, panel=panel, console=console
    )


def print_pretty(
    data: Any,
    *,
    title: str | None = None,
    title_align: AlignMethod = "left",
    box: bool = True,
    panel: bool = True,
    console: Console | None = None,
) -> None:
    r"""Render an arbitrary object to the console in a pretty format,
    optionally inside a Rich panel.

    Prints ``data`` using :class:`~rich.pretty.Pretty`, wrapped in a
    :class:`~rich.panel.Panel` unless ``panel`` is ``False``.  Uses the
    provided ``console`` if given, otherwise falls back to the shared
    instance.  Works with any Python object — dicts, lists, dataclasses,
    Pydantic models, and so on.

    Args:
        data: The object to render.  Passed directly to
            :class:`~rich.pretty.Pretty`, which handles formatting.
        title: Optional title displayed on the panel border.  Pass
            ``None`` (the default) to render a panel with no title.
            Ignored if ``panel`` is ``False``.
        title_align: Horizontal alignment of ``title`` along the panel
            border (``"left"``, ``"center"``, or ``"right"``).  Defaults
            to ``"left"``.  Ignored if ``panel`` is ``False``.
        box: If ``True`` (the default), render the panel with a visible
            rounded border.  If ``False``, render with a minimal
            (near-invisible) border.  Ignored if ``panel`` is ``False``.
        panel: If ``True`` (the default), wrap the rendered object in a
            :class:`~rich.panel.Panel`.  If ``False``, print it directly
            and ignore ``title``, ``title_align``, and ``box``.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Overrides the shared default for this
            invocation only.  Pass ``None`` (the default) to use the
            shared instance.

    Raises:
        MissingPackageError: If ``rich`` is not installed.

    Example:
        ```pycon
        >>> from io import StringIO
        >>> from rich.console import Console
        >>> from coola.display.rich import print_pretty
        >>> console = Console(file=StringIO(), width=20)
        >>> print_pretty({"key": "value"}, panel=False, console=console)
        >>> console.file.getvalue().strip()
        "{'key': 'value'}"
        >>> print_pretty({"key": "value"}, title="Demo")
        >>> print_pretty({"key": "value"}, title="Demo", box=False)
        >>> print_pretty({"key": "value"}, panel=False)

        ```
    """
    _print_in_panel(
        Pretty(data), title=title, title_align=title_align, box=box, panel=panel, console=console
    )
