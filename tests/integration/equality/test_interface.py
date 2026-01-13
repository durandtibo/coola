from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.equality import objects_are_equal
from coola.equality.tester import (
    SequenceEqualityTester,
    get_default_registry,
    register_equality_testers,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class CustomList(list): ...


def test_object_are_equal_custom_type() -> None:
    register_equality_testers({CustomList: SequenceEqualityTester()})
    assert objects_are_equal(CustomList([1, 2, 3]), CustomList([1, 2, 3]))
