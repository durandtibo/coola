"""Provide a shared Rich console and console rendering utilities."""

from __future__ import annotations

__all__ = ["print_markdown", "print_pretty"]

from typing import TYPE_CHECKING, Any

from rich import get_console
from rich.box import MINIMAL, ROUNDED
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

if TYPE_CHECKING:
    from rich.align import AlignMethod
    from rich.console import Console


def _truncate(content: str, max_length: int) -> tuple[str, bool]:
    """Truncate ``content`` to ``max_length`` characters, breaking on
    the nearest word boundary rather than mid-word when possible.

    Args:
        content: The text to truncate.
        max_length: The maximum number of characters to keep.

    Returns:
        A ``(truncated_content, was_truncated)`` tuple. If ``content``
        is already within ``max_length``, it is returned unchanged
        along with ``False``.
    """
    if len(content) <= max_length:
        return content, False

    cut = content[:max_length]
    last_space = cut.rfind(" ")
    # Only snap to the word boundary if it doesn't discard too much text.
    if last_space > max_length * 0.8:
        cut = cut[:last_space]
    return cut.rstrip(), True


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
            before truncating on a word boundary, with the omitted
            character count appended. Pass ``None`` (the default) to
            render ``msg`` in full.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Overrides the shared default for this
            invocation only.  Pass ``None`` (the default) to use the
            shared instance.

    Example:
        ```pycon
        >>> from coola.display.rich import print_markdown
        >>> print_markdown("**hello**", title="Demo")
        >>> print_markdown("**hello**", title="Demo", box=False)
        >>> print_markdown("**hello**", panel=False)

        ```
    """
    if max_length is not None:
        original_length = len(msg)
        msg, truncated = _truncate(msg, max_length)
        if truncated:
            omitted = original_length - len(msg)
            msg += f"\n\n*… ({omitted:,} more characters)*"
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

    Example:
        ```pycon
        >>> from coola.display.rich import print_pretty
        >>> print_pretty({"key": "value"}, title="Demo")
        >>> print_pretty({"key": "value"}, title="Demo", box=False)
        >>> print_pretty({"key": "value"}, panel=False)

        ```
    """
    _print_in_panel(
        Pretty(data), title=title, title_align=title_align, box=box, panel=panel, console=console
    )
