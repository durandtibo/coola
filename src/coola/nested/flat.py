r"""Contain functions to flat/unflat structured data."""

from __future__ import annotations

__all__ = ["from_flat_dict", "to_flat_dict"]

from collections.abc import Mapping, Sequence
from typing import Any, cast

from coola.validation import validate_not_empty

# ``str``, ``bytes`` and ``bytearray`` are sequences, but they are leaves.
_LEAF_SEQUENCES = (str, bytes, bytearray)


def from_flat_dict(
    data: dict[str, Any],
    *,
    separator: str = ".",
    restore_lists: bool = False,
    escape: bool = False,
) -> dict[str, Any]:
    r"""Return a nested dict from a flat dict produced by
    ``to_flat_dict``.

    Each key in ``data`` is split on ``separator`` to reconstruct the
    nesting depth.  By default the keys whose segments are decimal
    integers are **not** converted to lists; the output only contains
    ``dict`` objects, so the round-trip is lossless for the
    dict-of-dicts case and the integer keys are preserved as strings
    for the list-originated case.  Use ``restore_lists=True`` to
    rebuild the lists.

    Args:
        data: A flat dictionary whose keys use ``separator`` to encode
            nesting depth, as produced by ``to_flat_dict``.
        separator: The separator that was used when the flat dict was
            created.  Defaults to ``"."``.
        restore_lists: If ``True``, every ``dict`` whose keys are
            exactly the strings ``"0"``, ``"1"``, ..., ``"n-1"``
            (``n >= 1``) is converted to a ``list``. A ``dict`` that
            was originally built with such keys is converted too, so
            the original container type is not always recovered, and
            ``tuple`` objects come back as ``list`` objects.
            Defaults to ``False``.
        escape: If ``True``, a backslash escapes the separator (and
            the backslash itself) in the keys, as written by
            ``to_flat_dict(..., escape=True)``, so the keys of the
            original dict that contain the separator are restored.
            If ``False`` (default), every separator splits the key.

    Returns:
        A nested ``dict``. The root is always a ``dict``, even with
        ``restore_lists=True`` and the keys ``"0"``, ``"1"``, ...

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
        >>> # The lists can be restored
        >>> from_flat_dict(
        ...     {"module.0.0": 1, "module.0.1": 2, "module.1.bool": True, "str": "abc"},
        ...     restore_lists=True,
        ... )
        {'module': [[1, 2], {'bool': True}], 'str': 'abc'}

        ```
    """
    validate_not_empty(separator, name="separator")

    nested: dict[str, Any] = {}
    # Only the dicts created here can be extended. A dict given as a value belongs
    # to the caller and must be neither mutated nor extended.
    created: set[int] = {id(nested)}
    for flat_key, value in data.items():
        if not isinstance(flat_key, str):
            msg = f"Flat dict keys must be strings, got {flat_key!r} ({type(flat_key).__name__})"
            raise TypeError(msg)
        segments = _split_key(flat_key, separator) if escape else flat_key.split(separator)
        _set_nested(nested, segments, value, flat_key, separator, created=created)
    if restore_lists:
        # The root is always a dict, only its children can become lists.
        return {key: _restore_lists(value) for key, value in nested.items()}
    return nested


def _split_key(key: str, separator: str) -> list[str]:
    """Split *key* on the separators that are not escaped.

    A backslash followed by the separator or by a backslash gives the
    separator or the backslash; any other backslash is kept.
    """
    segments: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(key):
        if key[i] == "\\" and key.startswith("\\", i + 1):
            current.append("\\")
            i += 2
        elif key[i] == "\\" and key.startswith(separator, i + 1):
            current.append(separator)
            i += 1 + len(separator)
        elif key.startswith(separator, i):
            segments.append("".join(current))
            current = []
            i += len(separator)
        else:
            current.append(key[i])
            i += 1
    segments.append("".join(current))
    return segments


def _escape_key(key: str, separator: str) -> str:
    """Escape the backslashes and the separators of a key segment."""
    return key.replace("\\", "\\\\").replace(separator, "\\" + separator)


def _restore_lists(node: Any) -> Any:
    # A dict is converted to a list only if its keys are exactly "0", "1", ...,
    # "n-1": a gap or a different key means that it was not built from a list, and
    # converting it would lose the keys. The conversion is opt-in (``restore_lists``)
    # because a dict built with such keys looks the same as a list in the flat dict.
    if not isinstance(node, dict):
        return node
    restored = {key: _restore_lists(value) for key, value in node.items()}
    if restored and set(restored) == {str(i) for i in range(len(restored))}:
        return [restored[str(i)] for i in range(len(restored))]
    return restored


def _set_nested(
    target: dict[str, Any],
    segments: list[str],
    value: Any,
    original_key: str,
    separator: str,
    *,
    created: set[int],
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
        created: The ids of the dicts created by ``from_flat_dict``
            (updated in-place). A dict that is not in this set is a
            value given by the caller and is never extended.

    Raises:
        ValueError: If an interior segment collides with an existing
            scalar leaf, or the final segment collides with an existing
            nested dict.
    """
    current: dict[str, Any] = target
    for depth, segment in enumerate(segments[:-1]):
        if segment not in current:
            current[segment] = {}
            current = current[segment]
            created.add(id(current))
        else:
            existing: Any = current[segment]
            if isinstance(existing, dict) and id(existing) in created:
                current = cast("dict[str, Any]", existing)
            else:
                path_so_far = separator.join(segments[: depth + 1])
                msg = (
                    f"Key conflict: '{original_key}' tries to nest under "
                    f"'{path_so_far}', but that path is already a scalar leaf "
                    f"(value={existing!r}). Check that the flat dict was not "
                    "constructed from two incompatible sources."
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
    *,
    prefix: str | None = None,
    separator: str = ".",
    to_str: type | tuple[type, ...] | None = None,
    escape: bool = False,
) -> dict[str, Any]:
    r"""Return a flat representation of a nested structure as a dict
    using dotted keys.

    The keys of the nested dicts are joined with ``separator``, so by
    default a key that contains the separator is ambiguous:
    ``{"a.b": 1}`` and ``{"a": {"b": 1}}`` give the same flat dict, and
    ``from_flat_dict`` rebuilds the second one. Use ``escape=True`` to
    keep them apart.

    Args:
        data: The nested structure to flatten. Any ``Mapping`` or
            non-string, non-bytes ``Sequence`` (including ``range``) is
            recursed into; everything else, including the sets (they are
            not ordered, so they have no index), is treated as a leaf value.
        prefix: The prefix prepended to each key. ``None`` means no
            prefix; only valid when ``data`` is a mapping or sequence —
            a ``ValueError`` is raised if a bare scalar is passed
            without a prefix.
        separator: The separator used to join nested keys.
        to_str: A type or tuple of types that should be converted to
            their string representation instead of being recursed into.
            ``None`` (default) recurses into all supported containers.
        escape: If ``True``, a backslash is added before every
            separator and every backslash in the keys of the mappings
            (the ``prefix`` is used as is), so that
            ``from_flat_dict(..., escape=True)`` restores the keys that
            contain the separator. If ``False`` (default), the keys are
            joined as they are.

    Returns:
        A flat dictionary whose keys are the separator-joined
        paths to every leaf value.

    Raises:
        ValueError: If ``data`` is a scalar (non-mapping, non-sequence)
            and ``prefix`` is ``None``, since this would produce a
            ``None`` key, or if a container contains itself.

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
    flat: dict[str, Any] = {}
    _flatten_into(
        flat=flat,
        data=data,
        prefix=prefix,
        separator=separator,
        to_str=_normalize_to_str(to_str),
        ancestors=set(),
        escape=escape,
    )
    return flat


def _flatten_into(
    *,
    flat: dict[str, Any],
    data: object,
    prefix: str | None,
    separator: str,
    to_str: tuple[type, ...],
    ancestors: set[int],
    escape: bool,
) -> None:
    """Flatten *data* into the shared output dict *flat*.

    Every nested level writes directly into the same ``flat`` dict, so
    flattening is a single pass without intermediate per-level dicts.

    Args:
        flat: The output dict (mutated in-place).
        data: The value to flatten.
        prefix: The accumulated key path so far, or ``None`` at the
            root level.
        separator: The string used to join key segments.
        to_str: Normalised tuple of types that should be stringified
            rather than recursed into.
        ancestors: The ids of the containers being flattened, from the
            root to ``data``, used to detect the cycles.
        escape: If ``True``, the separators and the backslashes of the
            mapping keys are escaped.

    Raises:
        ValueError: If a leaf is reached with no key to store it under,
            or if a container contains itself.
    """
    if isinstance(data, to_str):
        if prefix is None:
            msg = (
                f"Cannot create a flat dict entry with a None key. "
                f"Provide a prefix when the top-level value is matched by to_str. "
                f"Got data={data!r}"
            )
            raise ValueError(msg)
        _set_leaf(flat, prefix, str(data))
    elif (
        isinstance(data, (Mapping, Sequence)) and not isinstance(data, _LEAF_SEQUENCES) and not data
    ):
        # Keep empty containers: they would otherwise vanish from the output.
        if prefix is not None:
            _set_leaf(flat, prefix, data)
    elif isinstance(data, Mapping):
        _check_no_cycle(data, ancestors)
        for key, value in cast("Mapping[Any, Any]", data).items():
            segment = _escape_key(str(key), separator) if escape else str(key)
            child = _build_key(prefix, segment, separator)
            _flatten_into(
                flat=flat,
                data=value,
                prefix=child,
                separator=separator,
                to_str=to_str,
                ancestors=ancestors,
                escape=escape,
            )
        ancestors.remove(id(data))
    # str, bytes and bytearray are Sequences, so they must be excluded before
    # the Sequence check to avoid iterating over individual characters or bytes.
    elif isinstance(data, Sequence) and not isinstance(data, _LEAF_SEQUENCES):
        _check_no_cycle(data, ancestors)
        for i, value in enumerate(cast("Sequence[Any]", data)):
            child = _build_key(prefix, str(i), separator)
            _flatten_into(
                flat=flat,
                data=value,
                prefix=child,
                separator=separator,
                to_str=to_str,
                ancestors=ancestors,
                escape=escape,
            )
        ancestors.remove(id(data))
    elif prefix is None:
        msg = (
            f"Cannot create a flat dict entry with a None key. "
            f"Provide a non-None prefix or pass a mapping/sequence as data. "
            f"Got data={data!r}"
        )
        raise ValueError(msg)
    else:
        _set_leaf(flat, prefix, data)


def _check_no_cycle(data: object, ancestors: set[int]) -> None:
    """Add *data* to the ancestors, refusing a container that contains
    itself.

    Raises:
        ValueError: If *data* is already one of its ancestors.
    """
    if id(data) in ancestors:
        msg = f"Cycle detected: an object of type {type(data)} contains itself"
        raise ValueError(msg)
    ancestors.add(id(data))


def _set_leaf(flat: dict[str, Any], key: str, value: Any) -> None:
    """Store *value* under *key*, refusing to overwrite an existing
    entry.

    Keys are converted with ``str()`` while flattening, so different
    keys (e.g. ``1`` and ``"1"``) can collide.
    """
    if key in flat:
        msg = (
            f"Key collision: several entries are flattened to the key {key!r}. "
            "Keys are converted to strings, so different keys can collide."
        )
        raise ValueError(msg)
    flat[key] = value


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
