r"""Contain some utility functions to remove and rename the keys of
nested mappings."""

from __future__ import annotations

__all__ = [
    "add_prefix_suffix_to_keys",
    "remove_keys_containing",
    "remove_keys_if",
    "remove_keys_starting_with",
]

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.nested.mapping import DuplicateKeyError
from coola.recursive import BaseTransformer, get_default_registry, recursive_apply
from coola.recursive.rebuild import rebuild_mapping
from coola.validation import validate_isinstance

if TYPE_CHECKING:
    from collections.abc import Callable

    from coola.recursive import TransformerRegistry
    from coola.recursive.base import Expansion

M = TypeVar("M", bound=Mapping[Any, Any])


class _KeyTransformer(InlineDisplayMixin, BaseTransformer[Mapping[Any, Any]]):
    r"""Transformer for mappings that drops and/or renames the keys.

    It is the only code of ``coola.nested`` that changes keys. The
    values are the children, so the engine of ``coola.recursive``
    walks them.

    Args:
        rename: A function applied to each key that is kept, or
            ``None`` to keep the keys.
        drop: A predicate applied to each key: the entries whose key
            satisfies it are dropped, or ``None`` to drop nothing.
    """

    def __init__(
        self,
        *,
        rename: Callable[[Any], Any] | None = None,
        drop: Callable[[Any], bool] | None = None,
    ) -> None:
        self._rename = rename
        self._drop = drop

    def _display_fields(self) -> dict[str, Any]:
        r"""Return the fields shown in the representation.

        Returns:
            The ``rename``, ``drop`` of the object.
        """
        return {"rename": self._rename, "drop": self._drop}

    def expand(self, data: Mapping[Any, Any]) -> Expansion | None:
        r"""Expand a mapping.

        Args:
            data: The mapping.

        Returns:
            The values of the kept entries, and the function that
                rebuilds the mapping with the renamed keys.

        Raises:
            DuplicateKeyError: If two entries have the same new key.
        """
        keys = [key for key in data if not (self._drop and self._drop(key))]
        new_keys = [self._rename(key) for key in keys] if self._rename else keys
        _check_no_duplicate(new_keys, data)
        return [data[key] for key in keys], lambda items: rebuild_mapping(
            data, dict(zip(new_keys, items, strict=True))
        )


def _check_no_duplicate(keys: list[Any], data: Mapping[Any, Any]) -> None:
    r"""Check that the keys of a mapping are all different.

    Args:
        keys: The new keys.
        data: The mapping whose keys are renamed, used in the message.

    Raises:
        DuplicateKeyError: If two entries have the same new key.
    """
    if len(set(keys)) != len(keys):
        msg = f"Renaming the keys of a {type(data).__name__} gives duplicate keys"
        raise DuplicateKeyError(msg)


def _identity(value: Any) -> Any:
    r"""Return the value unchanged.

    Args:
        value: The value.

    Returns:
        ``value``.
    """
    return value


def _apply(mapping: M, transformer: _KeyTransformer) -> M:
    r"""Apply a key transformer to every mapping of a nested structure.

    The registry is derived from the default one, so the containers are
    the ones of the other packages (dataclasses included) and the
    registrations of the user are followed.

    Args:
        mapping: The input mapping.
        transformer: The transformer used for the mappings.

    Returns:
        A new mapping of the same type as ``mapping``.

    Raises:
        TypeError: If ``mapping`` is not a mapping.
    """
    validate_isinstance(mapping, Mapping, name="mapping")
    registry: TransformerRegistry = get_default_registry().copy()
    # The mapping types of the default registry
    for data_type in (dict, Mapping, MappingProxyType):
        registry.register(data_type, transformer, exist_ok=True)
    return recursive_apply(mapping, _identity, registry=registry)


def add_prefix_suffix_to_keys(
    mapping: M,
    prefix: str = "",
    suffix: str = "",
    *,
    recursive: bool = False,
) -> M:
    r"""Add a prefix and/or a suffix to the keys of a mapping.

    Only keys of type ``str`` are renamed; every other key is kept
    as-is because a prefix/suffix cannot be concatenated to it. The
    type of every mapping is kept (an ``OrderedDict`` gives an
    ``OrderedDict``).

    Args:
        mapping: The input mapping.
        prefix: The prefix to prepend to the keys.
        suffix: The suffix to append to the keys.
        recursive: If ``True``, the prefix and suffix are also added
            to the keys of every nested mapping, including the mappings
            in the lists, the tuples and the dataclasses. If ``False``,
            only the keys of the top-level mapping are renamed.

    Returns:
        A new mapping of the same type with the renamed keys.

    Raises:
        TypeError: If ``mapping`` is not a mapping.
        ValueError: If a container contains itself.

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

    def rename(key: Any) -> Any:
        r"""Add the prefix and the suffix to a key.

        Args:
            key: The key.

        Returns:
            The new key if ``key`` is a ``str``, otherwise ``key``.
        """
        return f"{prefix}{key}{suffix}" if isinstance(key, str) else key

    if recursive:
        return _apply(mapping, _KeyTransformer(rename=rename))
    validate_isinstance(mapping, Mapping, name="mapping")
    new_keys = [rename(key) for key in mapping]
    _check_no_duplicate(new_keys, mapping)
    return rebuild_mapping(mapping, dict(zip(new_keys, mapping.values(), strict=True)))


def remove_keys_if(mapping: M, predicate: Callable[[Any], bool]) -> M:
    r"""Recursively remove the keys that satisfy a given predicate.

    The mapping is traversed with the engine of ``coola.recursive``
    and the containers of the other packages: mappings (``dict``,
    ``OrderedDict``, ...), sequences (``list``, ``tuple``, named
    tuples, ...), sets and dataclasses are entered, and the type of
    each container is kept. Every other value is kept as-is.

    Args:
        mapping: The original mapping.
        predicate: A function that takes a key and returns ``True``
            if the key (and its value) should be removed.

    Returns:
        A new mapping of the same type without the removed keys.

    Raises:
        TypeError: If ``mapping`` is not a mapping.
        ValueError: If a container contains itself.

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
    return _apply(mapping, _KeyTransformer(drop=predicate))


def remove_keys_containing(mapping: M, substring: str) -> M:
    r"""Recursively remove the keys that contain a given substring.

    Only the ``str`` keys are checked; the other keys are kept.

    Args:
        mapping: The original mapping.
        substring: The substring used to filter the keys.

    Returns:
        A new mapping of the same type without the removed keys.

    Raises:
        TypeError: If ``mapping`` is not a mapping.
        ValueError: If a container contains itself.

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


def remove_keys_starting_with(mapping: M, prefix: str) -> M:
    r"""Recursively remove the keys that start with a given prefix.

    Only the ``str`` keys are checked; the other keys are kept.

    Args:
        mapping: The original mapping.
        prefix: The prefix used to filter the keys.

    Returns:
        A new mapping of the same type without the removed keys.

    Raises:
        TypeError: If ``mapping`` is not a mapping.
        ValueError: If a container contains itself.

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
