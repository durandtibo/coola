from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.recursive import DefaultTransformer, IdentityTransformer, TransformerRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    return TransformerRegistry({object: DefaultTransformer()})


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
    assert objects_are_equal(
        IdentityTransformer().transform(data, func=func, registry=registry), data
    )


def test_identity_transformer_transform_returns_same_object(registry: TransformerRegistry) -> None:
    data = {"key": "value"}
    result = IdentityTransformer().transform(data, func=str, registry=registry)
    assert result is data


def test_identity_transformer_transform_ignores_predicate(registry: TransformerRegistry) -> None:
    assert (
        IdentityTransformer().transform("abc", func=lambda key: key == "abc", registry=registry)
        == "abc"
    )
