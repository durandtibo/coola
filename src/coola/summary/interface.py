r"""Define the public interface to summarize nested data."""

from __future__ import annotations

__all__ = [
    "get_default_registry",
    "get_default_summarizer_registry",
    "register_summarizers",
    "summarize",
]

import dataclasses
from collections import UserString, deque
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from coola.registry.default import make_default_registry_singleton
from coola.summary.config import SummaryConfig
from coola.summary.dataclass import DataclassSummarizer
from coola.summary.default import DefaultSummarizer
from coola.summary.mapping import MappingSummarizer
from coola.summary.registry import SummarizerRegistry
from coola.summary.sequence import SequenceSummarizer
from coola.summary.set import SetSummarizer
from coola.utils.imports import package_available

if TYPE_CHECKING:
    from coola.summary.base import Summarizer
    from coola.utils.singleton import LazySingleton

_DEFAULT_CONFIG = SummaryConfig()
_DEFAULT_LIMITS = (
    _DEFAULT_CONFIG.max_depth,
    _DEFAULT_CONFIG.max_items,
    _DEFAULT_CONFIG.max_characters,
)

# (names of the library in the module of the types, module with the summarizers)
_LAZY_SUMMARIZERS = (
    (("jax", "jaxlib"), "coola.summary.jax"),
    (("numpy",), "coola.summary.numpy"),
    (("pandas",), "coola.summary.pandas"),
    (("polars",), "coola.summary.polars"),
    (("pydantic",), "coola.summary.pydantic"),
    (("pyarrow",), "coola.summary.pyarrow"),
    (("torch",), "coola.summary.torch"),
    (("xarray",), "coola.summary.xarray"),
)


def summarize(
    data: object,
    *,
    max_depth: int = 1,
    max_items: int | None = 5,
    max_characters: int | None = 1000,
    show_data: bool = False,
    registry: SummarizerRegistry | None = None,
) -> str:
    r"""Create a summary string representation of nested data.

    The options must be passed by keyword.

    Args:
        data: Input data to summarize.
        max_depth: A display depth, not a limit that raises: the number
            of levels that are expanded. The structures below this
            depth are shown on one line (unlike the ``max_depth`` of
            ``objects_are_equal``, which raises an error for a deeper
            structure). Must be non-negative. Default is 1, which
            expands only the top level of nested structures.
        max_items: The maximum number of children (items, fields,
            columns) shown for a node. The other ones are announced
            with ``... (+N more)``. ``None`` shows all of them.
        max_characters: The maximum number of characters of a text.
            ``None`` does not cut the texts.
        show_data: If ``True``, the arrays, tensors and frames show a
            bounded ``repr`` of their values instead of their metadata.
        registry: Registry to resolve summarizers for nested data.
            If ``None``, the default registry is used.

    Returns:
        A formatted string summary of ``data``.

    Raises:
        TypeError: If ``max_depth``, ``max_items`` or ``max_characters``
            is not an ``int`` (a ``bool`` is rejected), or if
            ``show_data`` is not a ``bool``.
        ValueError: If ``max_depth``, ``max_items`` or
            ``max_characters`` is negative.

    Example:
        ```pycon
        >>> from coola.summary import summarize
        >>> print(summarize({"a": 1, "b": "abc"}))
        <class 'dict'> (length=2)
          (a): 1
          (b): abc
        >>> print(summarize(list(range(10)), max_items=3))
        <class 'list'> (length=10)
          (0): 0
          (1): 1
          (2): 2
          ... (+7 more)

        ```
    """
    if (
        (max_depth, max_items, max_characters) == _DEFAULT_LIMITS
        and (type(max_depth), type(max_items), type(max_characters)) == (int, int, int)
        and show_data is False
    ):
        config = _DEFAULT_CONFIG  # the config of most calls is built once
    else:
        config = SummaryConfig(
            max_depth=max_depth,
            max_items=max_items,
            max_characters=max_characters,
            show_data=show_data,
        )
    if registry is None:
        registry = get_default_registry()
    return registry.summarize(data, config)


def register_summarizers(
    mapping: Mapping[type, Summarizer[Any]],
    *,
    exist_ok: bool = False,
) -> None:
    """Register custom summarizers to the default global registry.

    This function extends the singleton registry returned by
    ``get_default_registry``.

    Args:
        mapping: Mapping of Python types to summarizer instances.
        exist_ok: If ``False``, raise an error when a type is already
            registered.

    Example:
        ```pycon
        >>> from coola.summary import register_summarizers, BaseSummarizer, SummaryConfig
        >>> class MyType:
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        >>> class MySummarizer(BaseSummarizer[MyType]):
        ...     def summarize(self, data: MyType, config: SummaryConfig, *, compact: bool) -> str:
        ...         return f"<MyType> value={data.value}"
        ...
        >>> register_summarizers({MyType: MySummarizer()})

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> SummarizerRegistry:
    """Get or create the default global registry with common Python
    types.

    Returns a singleton registry instance that is pre-configured with summarizers
    for Python's built-in types including sequences (list, tuple), mappings (dict),
    sets, and scalar types (int, float, str, bool).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A singleton ``SummarizerRegistry`` configured for common built-in
        scalar and container types.

    Notes:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new SummarizerRegistry instance directly.

    Example:
        ```pycon
        >>> from coola.summary import get_default_registry
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with common Python types
        >>> print(registry.summarize([1, 2, 3]))
        <class 'list'> (length=3)
          (0): 1
          (1): 2
          (2): 3
        >>> print(registry.summarize({"a": 1, "b": 2}))
        <class 'dict'> (length=2)
          (a): 1
          (b): 2

        ```
    """
    return _default_registry.get()


get_default_summarizer_registry = get_default_registry


def _register_default_summarizers(registry: SummarizerRegistry) -> None:
    """Register default summarizers for common Python types.

    This internal function sets up the standard type-to-summarizer mappings
    used by the default registry. It registers summarizers in a specific order
    to ensure proper inheritance handling via MRO.

    The registration strategy:
        - Scalar types use DefaultSummarizer (no recursion)
        - Strings use DefaultSummarizer to prevent character-by-character iteration
        - Sequences use SequenceSummarizer for summary list/tuple processing
        - Sets use SetSummarizer for summary set processing
        - Mappings use MappingSummarizer for summary dict processing

    Args:
        registry: The registry to populate with default summarizers

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    registry.register_many(_get_native_summarizers())
    # Dataclasses without their own summarizer
    registry.register_fallback(dataclasses.is_dataclass, DataclassSummarizer())
    # The summarizers of the optional libraries are loaded the first time the registry needs
    # a type of the library, so the libraries are not imported with ``coola.summary``.
    for names, module in _LAZY_SUMMARIZERS:
        if package_available(names[0]):
            registry.register_lazy(names, f"{module}:get_default_summarizers")


def _get_native_summarizers() -> dict[type, Summarizer[Any]]:
    r"""Get the native summarizers for common Python types.

    Returns:
        The native summarizers for common Python types.

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    default_summarizer = DefaultSummarizer()
    sequence_summarizer = SequenceSummarizer()
    set_summarizer = SetSummarizer()
    mapping_summarizer = MappingSummarizer()
    return {
        # Object is the catch-all base for unregistered types
        object: default_summarizer,
        # ``str`` and the numeric types are resolved to ``object`` with the MRO. ``UserString``
        # is a ``Sequence`` of ``UserString``: it must not be shown as a list of characters.
        UserString: default_summarizer,
        # Sequences - summary transformation preserving order
        list: sequence_summarizer,
        tuple: sequence_summarizer,
        deque: sequence_summarizer,
        Sequence: sequence_summarizer,
        # Sets - summary transformation without order
        set: set_summarizer,
        frozenset: set_summarizer,
        # Mappings - summary transformation of keys and values
        dict: mapping_summarizer,
        Mapping: mapping_summarizer,
        # A mapping proxy is only a virtual ``Mapping``, so it is not found with the MRO
        MappingProxyType: mapping_summarizer,
    }


_default_registry: LazySingleton[SummarizerRegistry] = make_default_registry_singleton(
    SummarizerRegistry, _register_default_summarizers, plugin_group="coola.summary.summarizers"
)
