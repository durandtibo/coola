from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality import objects_are_equal
from coola.equality.tester import (
    DefaultEqualityTester,
    get_default_registry as get_equality_registry,
)
from coola.hashing import get_default_registry as get_hasher_registry
from coola.iterator import get_default_registry as get_iterator_registry
from coola.recursive import get_default_registry as get_transformer_registry
from coola.registry import TypeNotRegisteredError, TypeRegistry
from coola.registry.default import reset_default_registries, temporary_default_registries
from coola.summary import get_default_registry as get_summarizer_registry
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING:
    from collections.abc import Callable

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


class Point:
    pass


###############################################
#     Tests for the default registries scope  #
###############################################


def test_temporary_default_registries_registers_a_type_inside_the_scope() -> None:
    with temporary_default_registries():
        get_equality_registry().register(Point, DefaultEqualityTester())
        assert get_equality_registry().has_equality_tester(Point)


def test_temporary_default_registries_does_not_leak_a_registration() -> None:
    with temporary_default_registries():
        get_equality_registry().register(Point, DefaultEqualityTester())
    assert not get_equality_registry().has_equality_tester(Point)


@numpy_available
def test_temporary_default_registries_restores_a_cleared_registry() -> None:
    with temporary_default_registries():
        get_equality_registry().clear()
    assert objects_are_equal(np.ones(2), np.ones(2))


def test_temporary_default_registries_cleared_registry_is_empty_inside_the_scope() -> None:
    with temporary_default_registries():
        get_equality_registry().clear()
        assert not get_equality_registry().has_equality_tester(int)


def test_temporary_default_registries_inside_the_scope_the_cleared_registry_raises() -> None:
    with temporary_default_registries():
        get_equality_registry().clear()
        with pytest.raises(TypeNotRegisteredError):
            objects_are_equal(Point(), Point())


def test_temporary_default_registries_restores_the_hasher_registry() -> None:
    with temporary_default_registries():
        get_hasher_registry().clear()
    assert len(get_hasher_registry().hash("abc")) == 64


def test_reset_default_registries_restores_a_cleared_registry() -> None:
    with temporary_default_registries():
        get_equality_registry().clear()
        reset_default_registries()
        assert get_equality_registry().has_equality_tester(int)


#####################################################
#     Tests for the isolated_default_registries     #
#####################################################


@pytest.mark.usefixtures("isolated_default_registries")
def test_isolated_default_registries_registers_a_type() -> None:
    get_equality_registry().register(Point, DefaultEqualityTester())
    assert get_equality_registry().has_equality_tester(Point)


def test_isolated_default_registries_does_not_leak_the_registration_to_the_next_test() -> None:
    """Test that the type registered by the previous test is not
    registered anymore."""
    assert not get_equality_registry().has_equality_tester(Point)


#####################################################
#     Tests for the copy of the default registries  #
#####################################################

DEFAULT_REGISTRIES = [
    pytest.param(get_equality_registry, id="equality"),
    pytest.param(get_hasher_registry, id="hashing"),
    pytest.param(get_iterator_registry, id="iterator"),
    pytest.param(get_summarizer_registry, id="summary"),
    pytest.param(get_transformer_registry, id="recursive"),
]


@pytest.mark.parametrize("get_registry", DEFAULT_REGISTRIES)
def test_default_registry_equals_its_copy(get_registry: Callable[[], TypeRegistry]) -> None:
    registry = get_registry()
    assert registry.copy().equal(registry)


@pytest.mark.parametrize("get_registry", DEFAULT_REGISTRIES)
def test_default_registry_copy_has_the_same_keys(get_registry: Callable[[], TypeRegistry]) -> None:
    registry = get_registry()
    assert list(registry.copy().keys()) == list(registry.keys())


@pytest.mark.parametrize("get_registry", DEFAULT_REGISTRIES)
def test_default_registry_copy_has_the_same_fallbacks(
    get_registry: Callable[[], TypeRegistry],
) -> None:
    registry = get_registry()
    assert list(registry.copy().fallbacks()) == list(registry.fallbacks())


@pytest.mark.parametrize("get_registry", DEFAULT_REGISTRIES)
def test_default_registry_copy_is_independent(get_registry: Callable[[], TypeRegistry]) -> None:
    registry = get_registry()
    registry.copy().register(Point, next(iter(registry.values())))
    assert not registry.has(Point)


@pytest.mark.parametrize("get_registry", DEFAULT_REGISTRIES)
def test_default_registry_repr_starts_with_the_class_name(
    get_registry: Callable[[], TypeRegistry],
) -> None:
    registry = get_registry()
    assert repr(registry).startswith(f"{type(registry).__name__}(")
