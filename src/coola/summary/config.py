r"""Define the options of a summary."""

from __future__ import annotations

__all__ = ["SummaryConfig"]

from dataclasses import dataclass

from coola.validation import validate_ge, validate_int, validate_isinstance


@dataclass(frozen=True, slots=True)
class SummaryConfig:
    r"""Define the options of a summary.

    The config is validated when it is created. It is immutable and
    hashable, so it can be shared by threads and calls.

    Args:
        max_depth: A display depth, not a limit that raises: the number
            of levels that are expanded. The nodes below this depth are
            shown on one line. Must be a non-negative ``int``.
        max_items: The maximum number of children (items, fields,
            columns) shown for a node. The other ones are announced
            with ``... (+N more)``. ``None`` shows all of them.
            Must be a non-negative ``int`` or ``None``.
        max_characters: The maximum number of characters of a text.
            ``None`` does not cut the texts. Must be a non-negative
            ``int`` or ``None``.
        show_data: If ``True``, the summarizers of arrays, tensors and
            frames show a bounded ``repr`` of the data instead of its
            metadata.

    Raises:
        TypeError: If ``max_depth``, ``max_items`` or ``max_characters``
            is not an ``int`` (a ``bool`` is rejected), or if
            ``show_data`` is not a ``bool``.
        ValueError: If ``max_depth``, ``max_items`` or
            ``max_characters`` is negative.

    Example:
        ```pycon
        >>> from coola.summary import SummaryConfig
        >>> SummaryConfig()
        SummaryConfig(max_depth=1, max_items=5, max_characters=1000, show_data=False)
        >>> SummaryConfig(max_depth=3, max_items=None)
        SummaryConfig(max_depth=3, max_items=None, max_characters=1000, show_data=False)

        ```
    """

    max_depth: int = 1
    max_items: int | None = 5
    max_characters: int | None = 1000
    show_data: bool = False

    def __post_init__(self) -> None:
        r"""Validate the options.

        Raises:
            TypeError: If ``max_depth``, ``max_items`` or
                ``max_characters`` is not an ``int`` (a ``bool`` is
                rejected), or if ``show_data`` is not a ``bool``.
            ValueError: If ``max_depth``, ``max_items`` or
                ``max_characters`` is negative.
        """
        # The plain valid values are checked first, because the config is built by each call of
        # ``summarize``.
        if type(self.max_depth) is not int or self.max_depth < 0:
            validate_int(self.max_depth, name="max_depth")
            validate_ge(self.max_depth, 0, name="max_depth")
        for name in ("max_items", "max_characters"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                validate_int(value, name=name)
                validate_ge(value, 0, name=name)
        if type(self.show_data) is not bool:
            validate_isinstance(self.show_data, bool, name="show_data")
