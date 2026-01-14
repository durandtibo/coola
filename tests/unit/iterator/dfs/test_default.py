from __future__ import annotations

from typing import Any

import pytest

from coola.equality import objects_are_equal
from coola.iterator.dfs import DefaultIterator, IteratorRegistry
from tests.unit.iterator.dfs.helpers import DEFAULT_SAMPLES

#####################################
#     Tests for DefaultIterator     #
#####################################


def test_default_iterator_repr() -> None:
    assert repr(DefaultIterator()) == "DefaultIterator()"


def test_default_iterator_str() -> None:
    assert str(DefaultIterator()) == "DefaultIterator()"


@pytest.mark.parametrize(("data", "expected"), DEFAULT_SAMPLES)
def test_default_iterator_iterate(data: Any, expected: Any) -> None:
    assert objects_are_equal(
        list(DefaultIterator().iterate(data, registry=IteratorRegistry())), expected
    )
