r"""Contain utilities for handlers."""

from __future__ import annotations

__all__ = [
    "AttributeName",
    "MaxDepthExceededError",
    "comparison_key",
    "is_exact_scalar_sequence",
]

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from coola.equality.config import EqualityConfig

# A NaN is replaced by this marker when ``equal_nan=True``.
_NAN: Any = object()

# The scalars for which ``==`` is exact: no NaN, no tolerance.
_EXACT_SCALAR_TYPES: frozenset[type] = frozenset({int, str, bytes, bool, type(None)})


class AttributeName(str):
    r"""Name of an attribute in the location of a difference.

    It is a ``str`` that marks a path component as an attribute (a
    dataclass field, for example) rather than a key or an index, so a
    path is formatted as ``['a'][2].x`` and not as ``['a'][2]['x']``.

    Example:
        ```pycon
        >>> from coola.equality._helpers.utils import AttributeName
        >>> AttributeName("x") == "x"
        True

        ```
    """

    __slots__ = ()


class MaxDepthExceededError(Exception):
    r"""Raised when the nesting of the compared objects exceeds the
    ``max_depth`` of the equality configuration."""


def comparison_key(value: Any, config: EqualityConfig) -> Any:
    r"""Return a hashable key that is equal for the equal values.

    This key is used to match the members of a set and the keys of a
    mapping. It contains the type of the value (``1``, ``1.0`` and
    ``True`` have the same hash and are equal with ``==``), and the NaNs
    are replaced by a marker if ``config.equal_nan`` is ``True``, also
    inside the tuples and frozensets.

    Args:
        value: The value (a member of a set or a key of a mapping).
        config: The equality configuration, which gives ``equal_nan``.

    Returns:
        A hashable key.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality._helpers.utils import comparison_key
        >>> comparison_key(float("nan"), EqualityConfig(equal_nan=True)) == comparison_key(
        ...     float("nan"), EqualityConfig(equal_nan=True)
        ... )
        True
        >>> comparison_key(1, EqualityConfig()) == comparison_key(True, EqualityConfig())
        False

        ```
    """
    if isinstance(value, float):
        if config.equal_nan and value != value:  # noqa: PLR0124
            return (float, _NAN)
    elif isinstance(value, tuple):
        return (type(value), tuple(comparison_key(v, config) for v in value))
    elif isinstance(value, frozenset):
        return (type(value), frozenset(Counter(comparison_key(v, config) for v in value).items()))
    return (type(value), value)


def is_exact_scalar_sequence(seq: Sequence[Any]) -> bool:
    r"""Indicate if ``seq`` is a list or tuple containing only exact
    built-in scalars.

    The exact scalars are the ``int``, ``str``, ``bytes``, ``bool`` and
    ``None`` values: ``==`` is exact for them (no NaN, no tolerance), so
    a sequence of exact scalars can be compared at once. The type of
    ``seq`` and the types of its items are compared exactly (no
    subclass), so a subclass of ``list`` is not accepted, and neither is
    a ``float`` (it can be NaN or compared with a tolerance).

    Args:
        seq: The sequence to check.

    Returns:
        ``True`` if ``seq`` is a ``list`` or a ``tuple`` and all its
            items have an exact scalar type, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.equality._helpers import is_exact_scalar_sequence
        >>> is_exact_scalar_sequence([1, "abc", None])
        True
        >>> is_exact_scalar_sequence([1, 2.0])
        False
        >>> is_exact_scalar_sequence([1, [2]])
        False

        ```
    """
    return type(seq) in (list, tuple) and set(map(type, seq)) <= _EXACT_SCALAR_TYPES
