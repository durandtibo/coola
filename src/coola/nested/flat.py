r"""Contain functions to flat/unflat structured data."""

from __future__ import annotations

__all__ = ["from_flat_dict", "to_flat_dict"]

from collections.abc import Mapping, Sequence
from typing import Any


def from_flat_dict(data: dict[str, Any], separator: str = ".") -> dict[str, Any]:
    r"""Return a nested dict from a flat dict produced by
    ``to_flat_dict``.

    Each key in ``data`` is split on ``separator`` to reconstruct the
    nesting depth.  Keys whose segments are all decimal integers are
    **not** converted to lists; the output is always a plain ``dict``
    so the round-trip is lossless for the dict-of-dicts case and
    predictable for the list-originated case (integer string keys are
    preserved as strings).

    Args:
        data: A flat dictionary whose keys use ``separator`` to encode
            nesting depth, as produced by :func:`to_flat_dict`.
        separator: The separator that was used when the flat dict was
            created.  Defaults to ``"."``.

    Returns:
        A nested ``dict``.

    Raises:
        ValueError: If ``separator`` is empty, which would make key
            splitting ambiguous.
        ValueError: If two keys in ``data`` imply conflicting types at
            the same path (e.g. one key treats a node as a dict and
            another treats it as a scalar leaf).

    Example:
        ```pycon
        >>> from coola.nested import from_flat_dict
        >>> from_flat_dict({"str": "def", "module.component.float": 3.5, "module.component.int": 2})
        {'str': 'def', 'module': {'component': {'float': 3.5, 'int': 2}}}
        >>> # Integer-string keys are preserved as strings
        >>> from_flat_dict({"module.0.0": 1, "module.0.1": 2, "module.1.bool": True, "str": "abc"})
        {'module': {'0': {'0': 1, '1': 2}, '1': {'bool': True}}, 'str': 'abc'}

        ```
    """
    if not separator:
        msg = "separator must be a non-empty string"
        raise ValueError(msg)

    nested: dict[str, Any] = {}
    for flat_key, value in data.items():
        segments = flat_key.split(separator)
        _set_nested(nested, segments, value, flat_key, separator)
    return nested


def _set_nested(
    target: dict[str, Any],
    segments: list[str],
    value: Any,
    original_key: str,
    separator: str,
) -> None:
    """Drill into *target* along *segments* and set the leaf to *value*.

    Iterates over all but the last segment, creating intermediate
    ``dict`` nodes as needed.  Raises ``ValueError`` on two kinds of
    conflict: a previous key already stored a scalar at an interior
    node that this key needs to nest under, or a previous key already
    stored a nested dict at the leaf position that this key needs to
    assign a scalar to.

    Args:
        target: The dict being built (mutated in-place).
        segments: The path components split from the original flat key.
        value: The leaf value to assign.
        original_key: The original flat key, used only for error messages.
        separator: The separator used to rejoin segments in error messages.

    Raises:
        ValueError: If an interior segment collides with an existing
            scalar leaf, or the final segment collides with an existing
            nested dict.
    """
    current = target
    for depth, segment in enumerate(segments[:-1]):
        existing = current.get(segment)
        if existing is None:
            current[segment] = {}
            current = current[segment]
        elif isinstance(existing, dict):
            current = existing
        else:
            path_so_far = separator.join(segments[: depth + 1])
            msg = (
                f"Key conflict: '{original_key}' tries to nest under "
                f"'{path_so_far}', but that path is already a scalar leaf "
                f"(value={existing!r}). Check that the flat dict was not "
                f"constructed from two incompatible sources."
            )
            raise ValueError(msg)

    leaf_key = segments[-1]
    if isinstance(current.get(leaf_key), dict):
        msg = (
            f"Key conflict: '{original_key}' tries to assign a scalar at "
            f"'{leaf_key}', but that key already holds a nested dict. "
            f"Check that the flat dict was not constructed from two "
            f"incompatible sources."
        )
        raise ValueError(msg)  # noqa: TRY004
    current[leaf_key] = value


def to_flat_dict(
    data: object,
    prefix: str | None = None,
    separator: str = ".",
    to_str: type | tuple[type, ...] | None = None,
) -> dict[str, Any]:
    r"""Return a flat representation of a nested structure as a dict
    using dotted keys.

    Args:
        data: The nested structure to flatten. Any :class:`Mapping` or
            non-string :class:`Sequence` is recursed into; everything
            else is treated as a leaf value.
        prefix: The prefix prepended to each key. ``None`` means no
            prefix; only valid when ``data`` is a mapping or sequence —
            a ``ValueError`` is raised if a bare scalar is passed
            without a prefix.
        separator: The separator used to join nested keys.
        to_str: A type or tuple of types that should be converted to
            their string representation instead of being recursed into.
            ``None`` (default) recurses into all supported containers.

    Returns:
        A flat dictionary whose keys are the separator-joined
        paths to every leaf value.

    Raises:
        ValueError: If ``data`` is a scalar (non-mapping, non-sequence)
            and ``prefix`` is ``None``, since this would produce a
            ``None`` key.

    Example:
        ```pycon
        >>> from coola.nested import to_flat_dict
        >>> data = {
        ...     "str": "def",
        ...     "module": {
        ...         "component": {
        ...             "float": 3.5,
        ...             "int": 2,
        ...         },
        ...     },
        ... }
        >>> to_flat_dict(data)
        {'str': 'def', 'module.component.float': 3.5, 'module.component.int': 2}

        >>> # Lists and tuples are also supported
        >>> data = {
        ...     "module": [[1, 2, 3], {"bool": True}],
        ...     "str": "abc",
        ... }
        >>> to_flat_dict(data)
        {'module.0.0': 1, 'module.0.1': 2, 'module.0.2': 3, 'module.1.bool': True, 'str': 'abc'}

        >>> # Use to_str to prevent recursion into specific types
        >>> to_flat_dict(data, to_str=list)
        {'module': "[[1, 2, 3], {'bool': True}]", 'str': 'abc'}

        ```
    """
    _to_str: tuple[type, ...] = _normalize_to_str(to_str)

    if isinstance(data, _to_str):
        if prefix is None:
            msg = (
                f"Cannot create a flat dict entry with a None key. "
                f"Provide a prefix when the top-level value is matched by to_str. "
                f"Got data={data!r}"
            )
            raise ValueError(msg)
        return {prefix: str(data)}

    if isinstance(data, Mapping):
        return _flatten_mapping(data, prefix, separator, _to_str)

    # str is a Sequence, so it must be excluded before the Sequence check
    # to avoid iterating over individual characters.
    if isinstance(data, Sequence) and not isinstance(data, str):
        return _flatten_sequence(data, prefix, separator, _to_str)

    # Scalar leaf
    if prefix is None:
        msg = (
            f"Cannot create a flat dict entry with a None key. "
            f"Provide a non-None prefix or pass a mapping/sequence as data. "
            f"Got data={data!r}"
        )
        raise ValueError(msg)
    return {prefix: data}


def _normalize_to_str(to_str: type | tuple[type, ...] | None) -> tuple[type, ...]:
    """Normalize the ``to_str`` argument to a (possibly empty) tuple of
    types.

    Accepts the three forms that the public API allows — ``None``, a
    single type, or an iterable of types — and always returns a
    ``tuple`` suitable for passing directly to ``isinstance``.

    Args:
        to_str: The raw ``to_str`` value passed by the caller.
            ``None`` means no types are converted to strings.

    Returns:
        A ``tuple`` of zero or more types.  An empty tuple causes
        ``isinstance(x, ())`` to always return ``False``, so nothing
        is converted to a string.
    """
    if to_str is None:
        return ()
    if isinstance(to_str, type):
        return (to_str,)
    return tuple(to_str)  # handles lists or other iterables defensively


def _build_key(prefix: str | None, child: str, separator: str) -> str:
    """Join a parent prefix and a child key segment with the separator.

    If ``prefix`` is ``None`` (i.e. we are at the root level with no
    enclosing key), the child segment is returned as-is.  Otherwise
    ``prefix``, ``separator``, and ``child`` are concatenated to form
    the full dotted path.

    Args:
        prefix: The accumulated key path so far, or ``None`` at the
            root level.
        child: The next key segment to append (a mapping key or a
            sequence index as a string).
        separator: The string used to join path segments.

    Returns:
        The combined key path as a string.
    """
    return f"{prefix}{separator}{child}" if prefix is not None else child


def _flatten_mapping(
    data: Mapping,
    prefix: str | None,
    separator: str,
    to_str: tuple[type, ...],
) -> dict[str, Any]:
    """Flatten a :class:`Mapping` into a flat ``dict`` with dotted keys.

    Iterates over every ``(key, value)`` pair, builds the child prefix
    by appending ``str(key)`` to the current ``prefix``, then
    delegates each value back to :func:`to_flat_dict` for further
    flattening.

    Args:
        data: The mapping to flatten.
        prefix: The accumulated key path so far, or ``None`` at the
            root level.
        separator: The string used to join key segments.
        to_str: Normalised tuple of types that should be stringified
            rather than recursed into.

    Returns:
        A flat dictionary.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        child_prefix = _build_key(prefix, str(key), separator)
        flat.update(to_flat_dict(value, prefix=child_prefix, separator=separator, to_str=to_str))
    return flat


def _flatten_sequence(
    data: Sequence,
    prefix: str | None,
    separator: str,
    to_str: tuple[type, ...],
) -> dict[str, Any]:
    """Flatten a non-string :class:`Sequence` into a flat ``dict`` with
    dotted keys.

    Iterates over every element by index, builds the child prefix by
    appending the string representation of the index to the current
    ``prefix``, then delegates each element back to
    :func:`to_flat_dict` for further flattening.

    Args:
        data: The sequence to flatten.  Must not be a ``str`` (strings
            are handled as scalar leaves by the caller).
        prefix: The accumulated key path so far, or ``None`` at the
            root level.
        separator: The string used to join key segments.
        to_str: Normalised tuple of types that should be stringified
            rather than recursed into.

    Returns:
        A flat dictionary.
    """
    flat: dict[str, Any] = {}
    for i, value in enumerate(data):
        child_prefix = _build_key(prefix, str(i), separator)
        flat.update(to_flat_dict(value, prefix=child_prefix, separator=separator, to_str=to_str))
    return flat
