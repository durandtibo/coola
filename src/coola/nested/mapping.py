r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = [
    "DuplicateKeyError",
    "flatten_mapping",
    "merge_mappings",
]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, assert_never

from coola.equality import objects_are_equal
from coola.validation import validate_in, validate_isinstance

if TYPE_CHECKING:
    from collections.abc import Iterable


class DuplicateKeyError(KeyError, ValueError):
    r"""Raised when a key cannot be added because it is a duplicate.

    It derives from ``KeyError`` (the historical exception) and from
    ``ValueError`` (the exception used for the other invalid inputs of
    this module), so both can be used to catch it.
    """

    def __str__(self) -> str:
        r"""Return the message without the quotes of ``KeyError``.

        Returns:
            The message of the error.
        """
        # ``KeyError`` would show the ``repr`` of the message (with quotes).
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


T = TypeVar("T")


def flatten_mapping(
    mapping: Mapping[Any, Mapping[Any, Any]],
    *,
    on_duplicate: Literal["raise", "first", "last", "prefix"] = "raise",
    always_prefix: bool = False,
    separator: str = ".",
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
                found, in place (the output follows the order of the
                input). This differs from
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
    validate_in(on_duplicate, ("raise", "first", "last", "prefix"), name="on_duplicate")

    # The entries are kept in a list so a key can be renamed in place: the output follows the
    # order of the input.
    entries: list[list[Any]] = []
    positions: dict[Any, int] = {}  # key of the output -> index of its entry
    origins: dict[Any, Any] = {}  # inner key -> outer key of the recorded occurrence
    seen: dict[Any, Any] = {}  # inner key -> recorded value, to check for real conflicts

    for outer_key, inner_mapping in mapping.items():
        validate_isinstance(inner_mapping, Mapping, name=f"value of the key {outer_key!r}")
        for inner_key, value in inner_mapping.items():
            prefixed_key = f"{outer_key}{separator}{inner_key}"

            if always_prefix:
                _add_entry(entries, positions, prefixed_key, value)
                continue

            if inner_key not in seen:
                seen[inner_key] = value
                origins[inner_key] = outer_key
                _add_entry(entries, positions, inner_key, value)
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
                entries[positions[inner_key]][1] = value
                continue
            if on_duplicate != "prefix":
                assert_never(on_duplicate)

            # Re-key the first occurrence too, the first time a real
            # conflict is found for this inner key.
            _prefix_first_occurrence(
                entries, positions, inner_key, f"{origins[inner_key]}{separator}{inner_key}"
            )
            _add_entry(entries, positions, prefixed_key, value)

    return dict(entries)


def _add_entry(entries: list[list[Any]], positions: dict[Any, int], key: Any, value: Any) -> None:
    r"""Add a new entry, refusing to overwrite an existing key.

    Different (outer key, inner key) pairs can give the same prefixed
    key, e.g. ``("a.b", "c")`` and ``("a", "b.c")``.

    Args:
        entries: The ``[key, value]`` entries of the flattened mapping,
            modified in-place.
        positions: The index in ``entries`` of each key, modified
            in-place.
        key: The key to add.
        value: The value to store.

    Raises:
        DuplicateKeyError: if ``key`` is already in ``entries``.
    """
    if key in positions:
        msg = f"Duplicate key found: {key!r} (it collides with another flattened key)"
        raise DuplicateKeyError(msg)
    positions[key] = len(entries)
    entries.append([key, value])


def _prefix_first_occurrence(
    entries: list[list[Any]], positions: dict[Any, int], key: Any, prefixed_key: str
) -> None:
    r"""Rename in place the first occurrence of ``key`` with its prefixed
    key, if it is still stored with its original key.

    Args:
        entries: The ``[key, value]`` entries of the flattened mapping,
            modified in-place.
        positions: The index in ``entries`` of each key, modified
            in-place.
        key: The original key.
        prefixed_key: The key with the prefix of its first occurrence.

    Raises:
        DuplicateKeyError: if ``prefixed_key`` is already in ``entries``.
    """
    if key not in positions:
        return
    if prefixed_key in positions:
        msg = f"Duplicate key found: {prefixed_key!r} (it collides with another flattened key)"
        raise DuplicateKeyError(msg)
    index = positions.pop(key)
    positions[prefixed_key] = index
    entries[index][0] = prefixed_key


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


def _free_suffixed_key(out: dict[Any, Any], input_keys: set[Any], key: Any, suffix: int) -> str:
    r"""Find the first key ``{key}_{n}`` with ``n >= suffix`` that is
    neither in ``out`` nor a key of an input mapping, so a generated key
    never overwrites or hides another key.

    Args:
        out: The merged mapping.
        input_keys: The keys of all the input mappings.
        key: The original key.
        suffix: The first suffix to try.

    Returns:
        The key with a suffix that is not used.
    """
    while f"{key}_{suffix}" in out or f"{key}_{suffix}" in input_keys:
        suffix += 1
    return f"{key}_{suffix}"


def merge_mappings(
    mappings: Iterable[Mapping[Any, Any]],
    *,
    on_duplicate: Literal["raise", "first", "last", "suffix"] = "raise",
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
        The keys of the output follow the order of the input. With
        ``on_duplicate='suffix'``, a generated key never takes the name of
        a key of an input mapping (whatever its position): the suffix is
        incremented until the key is free, so the result does not depend
        on the order of the mappings. Only the *later* occurrences
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
            more than once with different values.

    Example:
        ```pycon
        >>> from coola.nested import merge_mappings
        >>> merge_mappings([{"key1": 1, "key2": 2}, {"key2": 3, "key3": 4}], on_duplicate="last")
        {'key1': 1, 'key2': 3, 'key3': 4}

        ```
    """
    validate_in(on_duplicate, ("raise", "first", "last", "suffix"), name="on_duplicate")

    mappings = list(mappings)
    # The keys of the inputs: a generated key never takes the name of one of them, whatever the
    # position of its mapping, so the result does not depend on the order of the mappings.
    input_keys = (
        {key for mapping in mappings for key in mapping} if on_duplicate == "suffix" else set()
    )
    out: dict[Any, Any] = {}
    counts: dict[Any, int] = {}
    kept: dict[Any, list[Any]] = {}  # key -> the distinct values kept by the suffix strategy
    for mapping in mappings:
        for key, value in mapping.items():
            if key not in counts:
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
                out[_free_suffixed_key(out, input_keys, key, counts[key] - 1)] = value
            elif on_duplicate != "first":  # "first" keeps the existing value
                assert_never(on_duplicate)

    return out
