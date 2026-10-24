from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.equality import objects_are_equal
from coola.equality.tester import (
    DefaultEqualityTester,
    get_default_registry as get_equality_registry,
)
from coola.hashing import get_default_registry as get_hasher_registry
from coola.registry import TypeNotRegisteredError
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available
from coola.utils.singleton import reset_default_registries, temporary_default_registries

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
