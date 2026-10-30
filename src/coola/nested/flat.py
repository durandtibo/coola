r"""Contain functions to flat/unflat structured data."""

from __future__ import annotations

__all__ = ["from_flat_dict", "to_flat_dict"]

from typing import TYPE_CHECKING, Any

from coola.iterator import DefaultChildFinder, dfs_items, get_default_registry
from coola.recursive import IdentityTransformer, TransformerRegistry, recursive_apply
from coola.recursive.base import BaseTransformer
from coola.validation import validate_not_empty

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.iterator import ChildFinderRegistry
    from coola.recursive.base import Expansion

# ``str``, ``bytes`` and ``bytearray`` are sequences, but they are leaves.
_LEAF_SEQUENCES = (str, bytes, bytearray)


def from_flat_dict(
    data: Mapping[str, Any],
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
            ``tuple`` objects come back as ``list`` objects. Only the
            nodes built from the keys are converted: a ``dict`` given
            as a value is kept as it is. Defaults to ``False``.
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

    # Only the ``_Node`` created here can be extended. A dict given as a value is a plain
    # ``dict``: it belongs to the caller and must be neither mutated, extended nor converted.
    nested = _Root()
    for flat_key, value in data.items():
        if not isinstance(flat_key, str):
            msg = f"Flat dict keys must be strings, got {flat_key!r} ({type(flat_key).__name__})"
            raise TypeError(msg)
        segments = _split_key(flat_key, separator) if escape else flat_key.split(separator)
        _set_nested(nested, segments, value, flat_key, separator)
    return _freeze(nested, restore_lists=restore_lists)


class _Node(dict[str, Any]):
    r"""A dict created by ``from_flat_dict`` while it builds the result.

    A value given by the caller is a plain ``dict``, so
    ``isinstance(value, _Node)`` means that the dict belongs to the
    function.
    """


class _Root(_Node):
    r"""The root of the result, which is always a ``dict``."""


class _NodeTransformer(BaseTransformer[_Node]):
    r"""Transformer that turns a ``_Node`` into a ``dict``, or into a
    ``list`` when its keys are exactly ``"0"``, ..., ``"n-1"``.

    A dict is converted to a list only if its keys are exactly "0", "1",
    ..., "n-1": a gap or a different key means that it was not built
    from a list, and converting it would lose the keys. The conversion
    is opt-in (``restore_lists``) because a dict built with such keys
    looks the same as a list in the flat dict.

    Args:
        restore_lists: If ``True``, a node whose keys are ``"0"``,
            ..., ``"n-1"`` becomes a ``list``.
    """

    def __init__(self, *, restore_lists: bool) -> None:
        self._restore_lists = restore_lists

    def expand(self, data: _Node) -> Expansion | None:
        keys = list(data)
        restore = (
            self._restore_lists and bool(keys) and set(keys) == {str(i) for i in range(len(keys))}
        )
        if restore:
            return [data[str(i)] for i in range(len(keys))], list
        return [data[key] for key in keys], lambda items: dict(zip(keys, items, strict=True))


class _RootTransformer(_NodeTransformer):
    r"""Transformer of the root, which never becomes a list."""

    def __init__(self) -> None:
        super().__init__(restore_lists=False)


def _freeze(nested: _Root, *, restore_lists: bool) -> dict[str, Any]:
    r"""Replace the nodes built by ``from_flat_dict`` by ``dict`` and
    ``list``.

    The registry only knows the nodes: every other value (even a
    container given by the caller) is a leaf that is kept as it is.

    Args:
        nested: The root of the nodes.
        restore_lists: If ``True``, the nodes whose keys are ``"0"``,
            ..., ``"n-1"`` become lists.

    Returns:
        The nested ``dict``.
    """
    registry = TransformerRegistry(
        {
            object: IdentityTransformer(),
            _Node: _NodeTransformer(restore_lists=restore_lists),
            _Root: _RootTransformer(),
        }
    )
    # ``IdentityTransformer`` never applies the function, it is only required by the API
    return recursive_apply(nested, lambda value: value, registry=registry)


def _split_key(key: str, separator: str) -> list[str]:
    r"""Split *key* on the separators that are not escaped.

    A backslash followed by the separator or by a backslash gives the
    separator or the backslash; any other backslash is kept.

    Args:
        key: The flattened key.
        separator: The separator of the segments.

    Returns:
        The segments of the key.
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
    r"""Escape the backslashes and the separators of a key segment.

    Args:
        key: The key segment.
        separator: The separator of the segments.

    Returns:
        The escaped segment.
    """
    return key.replace("\\", "\\\\").replace(separator, "\\" + separator)


def _set_nested(
    target: _Node,
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
    current: _Node = target
    for depth, segment in enumerate(segments[:-1]):
        if segment not in current:
            current[segment] = current = _Node()
        else:
            existing: Any = current[segment]
            if isinstance(existing, _Node):
                current = existing
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
        data: The nested structure to flatten. The containers of
            ``coola.iterator`` are recursed into (mappings, sequences,
            dataclasses, ...); ``str``, ``bytes``, iterators and every
            unknown type are leaves. The sets are leaves too, whatever
            the registry: they are not ordered, so they have no index.
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
    to_str_types = _normalize_to_str(to_str)
    registry = _make_registry(to_str_types)
    flat: dict[str, Any] = {}
    for path, value in dfs_items(data, registry=registry, empty=True):
        if to_str_types and isinstance(value, to_str_types):
            leaf = str(value)
            from_to_str = True
        else:
            leaf = value
            from_to_str = False
        if not path and prefix is None:
            if from_to_str:
                msg = (
                    f"Cannot create a flat dict entry with a None key. "
                    f"Provide a prefix when the top-level value is matched by to_str. "
                    f"Got data={data!r}"
                )
                raise ValueError(msg)
            if registry.find_child_finder(type(value)).find_children(value) is None:
                msg = (
                    f"Cannot create a flat dict entry with a None key. "
                    f"Provide a non-None prefix or pass a mapping/sequence as data. "
                    f"Got data={data!r}"
                )
                raise ValueError(msg)
            continue  # an empty container at the root has no entry
        _set_leaf(flat, _build_key(prefix, path, separator, escape=escape), leaf)
    return flat


def _make_registry(to_str: tuple[type, ...]) -> ChildFinderRegistry:
    r"""Make the registry used to flatten the data.

    It is derived from the default registry of ``coola.iterator`` so the
    containers are the ones of the other packages. The sets (they have
    no index) and the types of ``to_str`` are leaves.

    Args:
        to_str: The types that are written with ``str()``.

    Returns:
        The registry.
    """
    registry = get_default_registry().copy()
    leaf = DefaultChildFinder()
    for data_type in (set, frozenset, *to_str):
        registry.register(data_type, leaf, exist_ok=True)
    if to_str:
        # A registered subclass of a ``to_str`` type is more specific, so it must be a leaf too.
        for data_type in list(registry.keys()):
            if issubclass(data_type, to_str):
                registry.register(data_type, leaf, exist_ok=True)
    return registry


def _set_leaf(flat: dict[str, Any], key: str, value: Any) -> None:
    r"""Store *value* under *key*, refusing to overwrite an existing
    entry.

    Keys are converted with ``str()`` while flattening, so different
    keys (e.g. ``1`` and ``"1"``) can collide.

    Args:
        flat: The flat dictionary to update.
        key: The flattened key.
        value: The value.

    Raises:
        ValueError: If ``key`` is already in ``flat``.
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


def _build_key(prefix: str | None, path: tuple[Any, ...], separator: str, *, escape: bool) -> str:
    """Join a prefix and the labels of a path with the separator.

    Args:
        prefix: The prefix of the key, or ``None`` for no prefix. It is
            used as it is.
        path: The labels of the containers crossed to reach the leaf: a
            key, an index or the name of a field.
        separator: The string used to join the segments.
        escape: If ``True``, the separators and the backslashes of the
            labels are escaped.

    Returns:
        The key.
    """
    segments = [_escape_key(str(label), separator) if escape else str(label) for label in path]
    if prefix is not None:
        segments.insert(0, prefix)
    return separator.join(segments)
