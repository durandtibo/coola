r"""Contain functions to flat/unflat structured data."""

from __future__ import annotations

__all__ = ["to_flat_dict"]

from typing import Any


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
