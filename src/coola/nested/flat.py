r"""Contain functions to flat/unflat structured data."""

from __future__ import annotations

__all__ = ["from_flat_dict", "to_flat_dict"]

from typing import Any


def from_flat_dict(
    data: dict[str, Any],
    separator: str = ".",
) -> dict[str, Any]:
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

    Args:
        target: The dict being built (mutated in-place).
        segments: The path components split from the original flat key.
        value: The leaf value to assign.
        original_key: The original flat key, used only for error messages.
        separator: The separator used to join segments in error messages.
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
            # A previous key assigned a scalar here; the two keys conflict.
            path_so_far = separator_from_segments(segments[: depth + 1], separator)
            msg = (
                f"Key conflict: '{original_key}' tries to nest under "
                f"'{path_so_far}', but that path is already a scalar leaf "
                f"(value={existing!r}). Check that the flat dict was not "
                f"constructed from two incompatible sources."
            )
            raise ValueError(msg)

    leaf_key = segments[-1]
    existing_leaf = current.get(leaf_key)
    if isinstance(existing_leaf, dict):
        msg = (
            f"Key conflict: '{original_key}' tries to assign a scalar at "
            f"'{leaf_key}', but that key already holds a nested dict. "
            f"Check that the flat dict was not constructed from two "
            f"incompatible sources."
        )
        raise ValueError(msg)
    current[leaf_key] = value


def separator_from_segments(segments: list[str], separator: str) -> str:
    """Reconstruct a dotted path from a list of segments (for error
    messages)."""
    return separator.join(segments)


def to_flat_dict(
    data: object,
    prefix: str | None = None,
    separator: str = ".",
    to_str: type | tuple[type, ...] | None = None,
) -> dict[str, Any]:
    r"""Return a flat representation of a nested structure as a dict
    using dotted keys.

    Args:
        data: The nested dict (or list/tuple) to flatten. Must be a
            dict when called without a prefix (i.e. at the top level).
        prefix: The prefix prepended to each key. ``None`` means no
            prefix; only valid when ``data`` is a dict or sequence —
            a ``ValueError`` is raised if a bare scalar is passed
            without a prefix.
        separator: The separator used to join nested keys.
        to_str: A type or tuple of types that should be converted to
            their string representation instead of being recursed into.
            ``None`` (default) recurses into all supported containers.

    Returns:
        A flat ``dict[str, Any]`` whose keys are the separator-joined
        paths to every leaf value.

    Raises:
        ValueError: If ``data`` is a scalar (non-dict, non-sequence)
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

    if isinstance(data, dict):
        return _flatten_mapping(data, prefix, separator, _to_str)

    if isinstance(data, (list, tuple)):
        return _flatten_sequence(data, prefix, separator, _to_str)

    # Scalar leaf
    if prefix is None:
        msg = (
            f"Cannot create a flat dict entry with a None key. "
            f"Provide a non-None prefix or pass a dict/sequence as data. "
            f"Got data={data!r}"
        )
        raise ValueError(msg)
    return {prefix: data}


def _normalize_to_str(to_str: type | tuple[type, ...] | None) -> tuple[type, ...]:
    """Normalize the ``to_str`` argument to a (possibly empty) tuple of
    types."""
    if to_str is None:
        return ()
    if isinstance(to_str, type):
        return (to_str,)
    return tuple(to_str)  # handles lists or other iterables defensively


def _build_key(prefix: str | None, child: str, separator: str) -> str:
    """Join a parent prefix and a child key segment with the
    separator."""
    return f"{prefix}{separator}{child}" if prefix is not None else child


def _flatten_mapping(
    data: dict,
    prefix: str | None,
    separator: str,
    to_str: tuple[type, ...],
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        child_prefix = _build_key(prefix, str(key), separator)
        flat.update(to_flat_dict(value, prefix=child_prefix, separator=separator, to_str=to_str))
    return flat


def _flatten_sequence(
    data: list | tuple,
    prefix: str | None,
    separator: str,
    to_str: tuple[type, ...],
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for i, value in enumerate(data):
        child_prefix = _build_key(prefix, str(i), separator)
        flat.update(to_flat_dict(value, prefix=child_prefix, separator=separator, to_str=to_str))
    return flat
