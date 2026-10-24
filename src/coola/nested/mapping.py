r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = [
    "DuplicateKeyError",
    "add_prefix_suffix_to_keys",
    "flatten_mapping",
    "get_first_value",
    "merge_mappings",
    "remove_keys_containing",
    "remove_keys_if",
    "remove_keys_starting_with",
]

from collections.abc import Mapping, Sequence, Set as AbstractSet
from typing import TYPE_CHECKING, Any, Literal, TypeVar, assert_never, cast

from coola.equality import objects_are_equal
from coola.recursive import (
    IdentityTransformer,
    KeyFilterTransformer,
    SequenceTransformer,
    SetTransformer,
    TransformerRegistry,
)
from coola.recursive.rebuild import rebuild_sequence

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class DuplicateKeyError(KeyError, ValueError):
    r"""Raised when a key cannot be added because it is a duplicate.

    It derives from ``KeyError`` (the historical exception) and from
    ``ValueError`` (the exception used for the other invalid inputs of
    this module), so both can be used to catch it.
    """

    def __str__(self) -> str:
        # ``KeyError`` would show the ``repr`` of the message (with quotes).
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


T = TypeVar("T")


def _check_is_mapping(key: Any, value: object) -> None:
    r"""Check that the value of a key is a mapping.

    Raises:
        TypeError: if ``value`` is not a mapping.
    """
    if not isinstance(value, Mapping):
        msg = f"The value of the key {key!r} must be a mapping, but got {type(value)!r}"
        raise TypeError(msg)


def flatten_mapping(
    mapping: Mapping[Any, Mapping[Any, Any]],
    on_duplicate: Literal["raise", "first", "last", "prefix"] = "raise",
    always_prefix: bool = False,
    separator: str = ".",
    *,
    equal_nan: bool = False,
) -> dict[Any, Any]:
    r"""Flatten a mapping of mappings into a single dict.

    Each outer key is used as a prefix for its inner keys. If an
    inner key appears in more than one outer mapping with the same
    value in every occurrence, the plain inner key is kept and
    ``on_duplicate`` is not triggered. ``on_duplicate`` only applies
    when an inner key appears more than once with different values.

    Args:
        mapping: The mapping of mappings to flatten.
        on_duplicate: The strategy used to manage duplicate inner
            keys that have different values across outer mappings.
            The valid values are:
            - ``'raise'``: raise an exception if an inner key
                appears more than once with different values.
            - ``'first'``: keep the value from the first occurrence,
                under the plain inner key.
            - ``'last'``: keep the value from the last occurrence,
                under the plain inner key.
            - ``'prefix'``: keep all the values, renaming every
                occurrence of a conflicting key as
                ``'{outer_key}{separator}{inner_key}'``, including
                the first occurrence. The result depends on the order of
                the mappings: a key is renamed only when a conflict is
                found. This differs from
                ``merge_mappings``'s ``on_duplicate='suffix'``
                strategy, which leaves the first occurrence under
                the plain key.
        always_prefix: If ``True``, every key in the output is named
            ``'{outer_key}{separator}{inner_key}'``, regardless of
            whether it is a duplicate. If ``False``, only keys
            involved in a genuine conflict are prefixed (only
            relevant when ``on_duplicate='prefix'``).
        separator: The separator used to join the outer and inner
            keys when prefixing.
        equal_nan: If ``True``, NaN values are considered equal, so
            they are not duplicates of each other. Defaults to
            ``False``.

    Returns:
        The flattened dict.

    Raises:
        ValueError: if ``on_duplicate`` is not a valid value.
        TypeError: if a value of ``mapping`` is not a mapping.
        DuplicateKeyError: if ``on_duplicate='raise'`` and an inner key
            appears more than once with different values, or if
            two different keys are flattened to the same key (for
            example ``{"a.b": {"c": 1}, "a": {"b.c": 2}}`` with
            ``always_prefix=True``).

    Example:
        ```pycon
        >>> from coola.nested import flatten_mapping
        >>> flatten_mapping(
        ...     {"module1": {"a": 1, "b": 2}, "module2": {"b": 3, "c": 4}},
        ...     on_duplicate="prefix",
        ... )
        {'a': 1, 'module1.b': 2, 'module2.b': 3, 'c': 4}

        ```
    """
    _validate_on_duplicate(on_duplicate, {"raise", "first", "last", "prefix"})

    out: dict[Any, Any] = {}
    origins: dict[Any, Any] = {}  # inner key -> outer key of the recorded occurrence
    seen: dict[Any, Any] = {}  # inner key -> recorded value, to check for real conflicts

    for outer_key, inner_mapping in mapping.items():
        _check_is_mapping(outer_key, inner_mapping)
        for inner_key, value in inner_mapping.items():
            prefixed_key = f"{outer_key}{separator}{inner_key}"

            if always_prefix:
                _add_key(out, prefixed_key, value)
                continue

            if inner_key not in seen:
                seen[inner_key] = value
                origins[inner_key] = outer_key
                _add_key(out, inner_key, value)
                continue

            if objects_are_equal(seen[inner_key], value, equal_nan=equal_nan):
                continue

            if on_duplicate == "raise":
                msg = f"Duplicate key found: {inner_key!r}"
                raise DuplicateKeyError(msg)
            if on_duplicate == "first":
                continue
            if on_duplicate == "last":
                seen[inner_key] = value
                out[inner_key] = value
                continue
            if on_duplicate != "prefix":
                assert_never(on_duplicate)

            # Re-key the first occurrence too, the first time a real
            # conflict is found for this inner key.
            _prefix_first_occurrence(out, inner_key, f"{origins[inner_key]}{separator}{inner_key}")
            _add_key(out, prefixed_key, value)

    return out


def _validate_on_duplicate(on_duplicate: str, strategies: set[str]) -> None:
    r"""Check that the strategy to handle the duplicate keys is valid.

    Args:
        on_duplicate: The strategy to check.
        strategies: The valid strategies.

    Raises:
        ValueError: if ``on_duplicate`` is not a valid strategy.
    """
    if on_duplicate not in strategies:
        msg = f"Incorrect on_duplicate value: {on_duplicate!r}. Valid values are {strategies}"
        raise ValueError(msg)


def _add_key(out: dict[Any, Any], key: Any, value: Any) -> None:
    r"""Add a new key, refusing to overwrite an existing entry.

    Different (outer key, inner key) pairs can give the same prefixed
    key, e.g. ``("a.b", "c")`` and ``("a", "b.c")``.

    Args:
        out: The flattened mapping, modified in-place.
        key: The key to add.
        value: The value to store.

    Raises:
        DuplicateKeyError: if ``key`` is already in ``out``.
    """
    if key in out:
        msg = f"Duplicate key found: {key!r} (it collides with another flattened key)"
        raise DuplicateKeyError(msg)
    out[key] = value


def _prefix_first_occurrence(out: dict[Any, Any], key: Any, prefixed_key: str) -> None:
    r"""Re-key the first occurrence of ``key`` with its prefixed key, if
    it is still stored with its original key.

    Args:
        out: The flattened mapping, modified in-place.
        key: The original key.
        prefixed_key: The key with the prefix of its first occurrence.
    """
    if key in out:
        value = out.pop(key)
        _add_key(out, prefixed_key, value)


def _is_duplicate_value(value: Any, kept: list[Any], equal_nan: bool) -> bool:
    r"""Indicate if a value is equal to one of the kept values.

    Args:
        value: The new value.
        kept: The values already kept for the key.
        equal_nan: If ``True``, NaN values are considered equal.

    Returns:
        ``True`` if ``value`` is equal to one of ``kept``.
    """
    return any(objects_are_equal(v, value, equal_nan=equal_nan) for v in kept)


def _free_suffixed_key(out: dict[Any, Any], key: Any, suffix: int) -> str:
    r"""Find the first key ``{key}_{n}`` with ``n >= suffix`` that is not
    in ``out``, so an existing key is never overwritten.

    Args:
        out: The merged mapping.
        key: The original key.
        suffix: The first suffix to try.

    Returns:
        The key with a suffix that is not used.
    """
    while f"{key}_{suffix}" in out:
        suffix += 1
    return f"{key}_{suffix}"


def add_prefix_suffix_to_keys(
    mapping: Mapping[Any, Any],
    prefix: str = "",
    suffix: str = "",
    recursive: bool = False,
) -> dict[Any, Any]:
    r"""Add a prefix and/or a suffix to the keys of a mapping.

    Only keys of type ``str`` are renamed; every other key is kept
    as-is because a prefix/suffix cannot be concatenated to it.

    Args:
        mapping: The input mapping.
        prefix: The prefix to prepend to the keys.
        suffix: The suffix to append to the keys.
        recursive: If ``True``, the prefix and suffix are also added
            to the keys of every nested mapping, including the mappings
            in the lists and the tuples. If ``False``, only
            the keys of the top-level mapping are renamed.

    Returns:
        A new dict with the renamed keys.

    Example:
        ```pycon
        >>> from coola.nested import add_prefix_suffix_to_keys
        >>> add_prefix_suffix_to_keys({"key1": 1, "key2": 2}, prefix="prefix_")
        {'prefix_key1': 1, 'prefix_key2': 2}
        >>> add_prefix_suffix_to_keys(
        ...     {"key1": 1, "key2": {"key3": 3}}, suffix="_suffix", recursive=True
        ... )
        {'key1_suffix': 1, 'key2_suffix': {'key3_suffix': 3}}

        ```
    """
    return {
        (f"{prefix}{key}{suffix}" if isinstance(key, str) else key): (
            _rename_nested_keys(value, prefix, suffix) if recursive else value
        )
        for key, value in mapping.items()
    }


def _rename_nested_keys(data: Any, prefix: str, suffix: str) -> Any:
    r"""Add a prefix and a suffix to the keys of the mappings in a value.

    The mappings are found in the value itself and in the sequences
    (``list``, ``tuple``, ...) it contains. Any other value is returned
    as-is.

    Args:
        data: The value to process.
        prefix: The prefix to prepend to the keys.
        suffix: The suffix to append to the keys.

    Returns:
        The value with the renamed keys.
    """
    if isinstance(data, Mapping):
        return add_prefix_suffix_to_keys(
            cast("Mapping[Any, Any]", data), prefix=prefix, suffix=suffix, recursive=True
        )
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return rebuild_sequence(
            data,
            [_rename_nested_keys(item, prefix, suffix) for item in cast("Sequence[Any]", data)],
        )
    return data


def get_first_value(data: Mapping[Any, T]) -> T:
    r"""Get the first value of a mapping.

    Args:
        data: The input mapping.

    Returns:
        The first value in the mapping.

    Raises:
        ValueError: if the mapping is empty.

    Example:
        ```pycon
        >>> from coola.nested import get_first_value
        >>> get_first_value({"key1": 1, "key2": 2})
        1

        ```
    """
    if not data:
        msg = "First value cannot be returned because the mapping is empty"
        raise ValueError(msg)
    return next(iter(data.values()))


def merge_mappings(
    mappings: Iterable[Mapping[Any, Any]],
    on_duplicate: Literal["raise", "first", "last", "suffix"] = "raise",
    *,
    equal_nan: bool = False,
) -> dict[Any, Any]:
    r"""Merge an iterable of mappings into a single dict.

    If a key appears more than once with the same value in every
    occurrence, the value is kept as-is and ``on_duplicate`` is not
    triggered. ``on_duplicate`` only applies when a key appears more
    than once with different values.

    Args:
        mappings: The mappings to merge.
        on_duplicate: The strategy used to manage duplicate keys that
            have different values across mappings.
            The valid values are:
            - ``'raise'``: raise an exception if a key appears
                more than once with different values.
            - ``'first'``: keep the value from the first occurrence.
            - ``'last'``: keep the value from the last occurrence.
            - ``'suffix'``: keep all the distinct values by adding a
                ``'_n'`` suffix to the key name, where ``n`` is
                incremented for each new distinct value. A value that
                is equal to one already kept for the key is dropped.
        equal_nan: If ``True``, NaN values are considered equal, so
            they are not duplicates of each other. Defaults to
            ``False``.

    Note:
        With ``on_duplicate='suffix'``, only the *later* occurrences
        of a conflicting key are renamed (``'{key}_1'``,
        ``'{key}_2'``, ...); the first occurrence keeps the plain
        key. This differs from ``flatten_mapping``'s
        ``on_duplicate='prefix'`` strategy, which renames *every*
        occurrence, including the first, once a genuine conflict is
        detected for a key.

    Returns:
        The merged dict.

    Raises:
        ValueError: if ``on_duplicate`` is not a valid value.
        DuplicateKeyError: if ``on_duplicate='raise'`` and a key appears
            more than once with different values, or if
            ``on_duplicate='suffix'`` and a key of an input mapping
            is equal to a key generated by the suffix strategy.

    Example:
        ```pycon
        >>> from coola.nested import merge_mappings
        >>> merge_mappings([{"key1": 1, "key2": 2}, {"key2": 3, "key3": 4}], on_duplicate="last")
        {'key1': 1, 'key2': 3, 'key3': 4}

        ```
    """
    _validate_on_duplicate(on_duplicate, {"raise", "first", "last", "suffix"})

    out: dict[Any, Any] = {}
    counts: dict[Any, int] = {}
    kept: dict[Any, list[Any]] = {}  # key -> the distinct values kept by the suffix strategy
    for mapping in mappings:
        for key, value in mapping.items():
            if key not in counts:
                if key in out:
                    # The key was generated by the ``suffix`` strategy.
                    msg = f"Duplicate key found: {key!r} (it collides with a suffixed key)"
                    raise DuplicateKeyError(msg)
                counts[key] = 1
                out[key] = value
                kept[key] = [value]
                continue

            if _is_duplicate_value(
                value, kept[key] if on_duplicate == "suffix" else [out[key]], equal_nan
            ):
                # Same value repeated: not a real conflict, keep as-is.
                continue
            kept[key].append(value)

            counts[key] += 1
            if on_duplicate == "raise":
                msg = f"Duplicate key found: {key!r}"
                raise DuplicateKeyError(msg)
            if on_duplicate == "last":
                out[key] = value
            elif on_duplicate == "suffix":
                out[_free_suffixed_key(out, key, counts[key] - 1)] = value
            elif on_duplicate != "first":  # "first" keeps the existing value
                assert_never(on_duplicate)

    return out


def _make_key_filter_registry(key_predicate: Callable[[Any], bool]) -> TransformerRegistry:
    """Build the registry used by ``remove_keys_if`` to recurse through
    arbitrary nested containers.

    Reuses the same type-dispatch machinery as ``coola.recursive`` so
    any registered ``Mapping``, ``Sequence`` (list, tuple, namedtuple,
    custom subclasses, ...) or ``Set``/``frozenset`` is traversed and
    rebuilt with its original type, while every other value (including
    ``str``) is left untouched.

    The registry is built for each call because the key predicate is
    stored in the transformer of the mappings (building it takes about
    a microsecond).

    Args:
        key_predicate: The function that selects the keys to remove.

    Returns:
        A registry mapping container types to the transformers used
        to recurse into them.
    """
    identity = IdentityTransformer()
    sequence = SequenceTransformer()
    key_filter = KeyFilterTransformer(key_predicate=key_predicate)
    return TransformerRegistry(
        {
            object: identity,
            str: identity,
            list: sequence,
            tuple: sequence,
            Sequence: sequence,
            set: SetTransformer(),
            frozenset: SetTransformer(),
            AbstractSet: SetTransformer(),
            dict: key_filter,
            Mapping: key_filter,
        }
    )


def remove_keys_if(mapping: Mapping[Any, Any], predicate: Callable[[Any], bool]) -> dict[Any, Any]:
    r"""Recursively remove the keys that satisfy a given predicate.

    The mapping is traversed recursively using the same type-dispatch
    machinery as ``coola.recursive.recursive_apply``, so it works
    on arbitrary nested combinations of mappings (``dict``,
    ``OrderedDict``, ...), sequences (``list``, ``tuple``, named
    tuples, ...), and sets/frozensets, not just plain ``dict``/
    ``list``/``tuple``. Every other value is kept as-is.

    Args:
        mapping: The original mapping.
        predicate: A function that takes a key and returns ``True``
            if the key (and its value) should be removed.

    Returns:
        A new dict without the removed keys.

    Example:
        ```pycon
        >>> from coola.nested import remove_keys_if
        >>> remove_keys_if(
        ...     {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6},
        ...     lambda key: isinstance(key, str) and "key" in key,
        ... )
        {'abc': 3, 1: 5, (2, 3): 6}
        >>> remove_keys_if(
        ...     {"abc": {"key": 1, "abc": 2}, "list": [{"key": 1, "abc": 2}]},
        ...     lambda key: isinstance(key, str) and "key" in key,
        ... )
        {'abc': {'abc': 2}, 'list': [{'abc': 2}]}

        ```
    """
    return _make_key_filter_registry(predicate).transform(mapping, lambda value: value)


def remove_keys_containing(mapping: Mapping[Any, Any], substring: str) -> dict[Any, Any]:
    r"""Recursively remove the keys that contain a given substring.

    Args:
        mapping: The original mapping.
        substring: The substring used to filter the keys.

    Returns:
        A new dict without the removed keys.

    Example:
        ```pycon
        >>> from coola.nested import remove_keys_containing
        >>> remove_keys_containing(
        ...     {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6},
        ...     "key",
        ... )
        {'abc': 3, 1: 5, (2, 3): 6}
        >>> remove_keys_containing(
        ...     {"abc": {"key": 1, "abc": 2}, "list": [{"key": 1, "abc": 2}]},
        ...     "key",
        ... )
        {'abc': {'abc': 2}, 'list': [{'abc': 2}]}

        ```
    """
    return remove_keys_if(mapping, lambda key: isinstance(key, str) and substring in key)


def remove_keys_starting_with(mapping: Mapping[Any, Any], prefix: str) -> dict[Any, Any]:
    r"""Recursively remove the keys that start with a given prefix.

    Args:
        mapping: The original mapping.
        prefix: The prefix used to filter the keys.

    Returns:
        A new dict without the removed keys.

    Example:
        ```pycon
        >>> from coola.nested import remove_keys_starting_with
        >>> remove_keys_starting_with(
        ...     {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6},
        ...     "key",
        ... )
        {'abc': 3, 'abc.key': 4, 1: 5, (2, 3): 6}

        ```
    """
    return remove_keys_if(mapping, lambda key: isinstance(key, str) and key.startswith(prefix))
