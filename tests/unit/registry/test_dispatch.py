from __future__ import annotations

import pytest

from coola.equality.tester.registry import EqualityTesterRegistry
from coola.hashing.registry import HasherRegistry
from coola.iterator.bfs.registry import ChildFinderRegistry
from coola.iterator.dfs.registry import IteratorRegistry
from coola.recursive.registry import TransformerRegistry
from coola.registry.dispatch import BaseTypeDispatchRegistry
from coola.summary.registry import SummarizerRegistry

##############################################
#     Tests for BaseTypeDispatchRegistry     #
##############################################


def test_base_type_dispatch_registry_init_empty() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    assert len(registry._state) == 0


def test_base_type_dispatch_registry_init_state_is_copied() -> None:
    state = {int: "int_handler"}
    registry = BaseTypeDispatchRegistry[str](state)
    registry.register(float, "float_handler")
    assert float not in state


def test_base_type_dispatch_registry_register_and_find() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register(int, "int_handler")
    assert registry.find(int) == "int_handler"


def test_base_type_dispatch_registry_register_exist_ok_false_raises_error() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register(int, "new_handler")


def test_base_type_dispatch_registry_register_exist_ok_true_overwrites() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    registry.register(int, "new_handler", exist_ok=True)
    assert registry.find(int) == "new_handler"


def test_base_type_dispatch_registry_register_many() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register_many({int: "int_handler", str: "str_handler"})
    assert registry.find(int) == "int_handler"
    assert registry.find(str) == "str_handler"


def test_base_type_dispatch_registry_has_true() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    assert registry.has(int)


def test_base_type_dispatch_registry_has_false() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    assert not registry.has(int)


def test_base_type_dispatch_registry_has_false_for_mro_only_match() -> None:
    # `has` only checks direct registration, not MRO resolution.
    registry = BaseTypeDispatchRegistry[str]({object: "default_handler"})
    assert not registry.has(int)


def test_base_type_dispatch_registry_find_resolves_via_mro() -> None:
    registry = BaseTypeDispatchRegistry[str]({object: "default_handler"})
    assert registry.find(int) == "default_handler"


def test_base_type_dispatch_registry_find_missing_raises_error() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    with pytest.raises(KeyError):
        registry.find(int)


def test_base_type_dispatch_registry_repr() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    assert repr(registry) == (
        "BaseTypeDispatchRegistry(\n"
        "  (state): TypeRegistry(\n"
        "      (<class 'int'>): int_handler\n"
        "    )\n"
        ")"
    )


@pytest.mark.parametrize(
    "registry_cls",
    [
        EqualityTesterRegistry,
        HasherRegistry,
        TransformerRegistry,
        SummarizerRegistry,
        ChildFinderRegistry,
        IteratorRegistry,
    ],
)
def test_registries_are_subclasses_of_base_type_dispatch_registry(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that each of the five (plus one) registry wrappers reuses
    the shared base implementation instead of hand-rolling
    register/register_many/has/find."""
    assert issubclass(registry_cls, BaseTypeDispatchRegistry)


@pytest.mark.parametrize(
    "registry_cls",
    [
        EqualityTesterRegistry,
        HasherRegistry,
        TransformerRegistry,
        SummarizerRegistry,
        ChildFinderRegistry,
        IteratorRegistry,
    ],
)
def test_registries_register_many_shares_base_implementation(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that ``register_many`` on a concrete registry actually
    delegates to the shared base implementation (registers all entries,
    is exist_ok-aware, does not partially mutate state on failure)."""

    class Handler:
        def __repr__(self) -> str:
            return "Handler()"

    registry = registry_cls()
    h1, h2 = Handler(), Handler()
    registry.register_many({int: h1, str: h2})
    assert registry.has(int)
    assert registry.has(str)

    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many({int: Handler()})

    # exist_ok=True overwrites without error
    h3 = Handler()
    registry.register_many({int: h3}, exist_ok=True)
    assert registry.find(int) is h3


##############################################################
#     Tests for de-duplicated has_<x>/find_<x> docstrings    #
##############################################################

# Method-name pairs for the "thin wrapper" `has_<x>`/`find_<x>` methods that
# each concrete registry exposes around the shared `has`/`find` from
# `BaseTypeDispatchRegistry`. See design/code-review-findings.md, section 3
# ("Type-lookup docstring/example blocks are copy-pasted nearly verbatim").
_WRAPPER_METHOD_NAMES = [
    ("has_equality_tester", "find_equality_tester"),
    ("has_hasher", "find_hasher"),
    ("has_transformer", "find_transformer"),
    ("has_summarizer", "find_summarizer"),
    ("has_child_finder", "find_child_finder"),
    ("has_iterator", "find_iterator"),
]


@pytest.mark.parametrize(
    ("registry_cls", "names"),
    list(
        zip(
            [
                EqualityTesterRegistry,
                HasherRegistry,
                TransformerRegistry,
                SummarizerRegistry,
                ChildFinderRegistry,
                IteratorRegistry,
            ],
            _WRAPPER_METHOD_NAMES,
            strict=True,
        )
    ),
)
def test_registry_wrapper_docstrings_reference_base_class(
    registry_cls: type[BaseTypeDispatchRegistry], names: tuple[str, str]
) -> None:
    """Test that each ``has_<x>``/``find_<x>`` wrapper documents its
    behavior by pointing at ``BaseTypeDispatchRegistry.has``/``.find``
    instead of duplicating the full behavioral description, so a future
    correction only needs to be made once, in the base class."""
    has_name, find_name = names
    has_doc = getattr(registry_cls, has_name).__doc__
    find_doc = getattr(registry_cls, find_name).__doc__
    assert "BaseTypeDispatchRegistry.has" in has_doc
    assert "BaseTypeDispatchRegistry.find" in find_doc


@pytest.mark.parametrize(
    "registry_cls",
    [
        EqualityTesterRegistry,
        HasherRegistry,
        TransformerRegistry,
        SummarizerRegistry,
        ChildFinderRegistry,
        IteratorRegistry,
    ],
)
def test_registry_wrapper_docstrings_do_not_duplicate_mro_prose(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that the ``find_<x>`` wrapper docstrings no longer duplicate
    the long "uses the Method Resolution Order..." behavioral
    description that used to be copy-pasted (with minor drift) into
    every registry; that description now lives only on
    ``BaseTypeDispatchRegistry.find``."""
    find_name = next(name for _, name in _WRAPPER_METHOD_NAMES if hasattr(registry_cls, name))
    doc = getattr(registry_cls, find_name).__doc__
    assert "Method Resolution Order" not in doc
    assert "BaseTypeDispatchRegistry.find" in doc


def test_base_type_dispatch_registry_find_docstring_documents_caching() -> None:
    """Test that the caching behavior of ``find`` is documented once, in
    the base class, rather than repeated (and drifting, e.g. the stale
    "LRU cache (256 entries)" claim flagged in the review) across each
    concrete registry's ``find_<x>`` wrapper."""
    assert "cached" in BaseTypeDispatchRegistry.find.__doc__
