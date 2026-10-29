from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.recursive import IdentityTransformer, TransformerRegistry
from tests.unit.recursive.helpers import expand_not_none

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    return TransformerRegistry({object: IdentityTransformer()})


@pytest.mark.parametrize(
    ("data", "func"),
    [
        pytest.param(42, str, id="int"),
        pytest.param("hello", str.upper, id="str"),
        pytest.param(3.14, round, id="float"),
        pytest.param(None, str, id="none"),
        pytest.param([1, 2, 3], str, id="list"),
        pytest.param({"a": 1}, str, id="dict"),
    ],
)
def test_identity_transformer_transform_ignores_func(
    data: Any, func: Callable, registry: TransformerRegistry
) -> None:
    assert objects_are_equal(registry.transform(data, func), data)


def test_identity_transformer_transform_returns_same_object(registry: TransformerRegistry) -> None:
    data = {"key": "value"}
    result = registry.transform(data, str)
    assert result is data


def test_identity_transformer_transform_ignores_predicate(registry: TransformerRegistry) -> None:
    assert registry.transform("abc", lambda key: key == "abc") == "abc"


def test_identity_transformer_expand_has_no_children() -> None:
    children, _ = expand_not_none(IdentityTransformer(), [1, 2])
    assert children == ()


def test_identity_transformer_expand_rebuild_returns_the_data() -> None:
    data = [1, 2]
    _, rebuild = expand_not_none(IdentityTransformer(), data)
    assert rebuild([]) is data
