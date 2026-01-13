from __future__ import annotations

from typing import Any

import pytest

from coola.equality import objects_are_equal
from coola.recursive import (
    ConditionalTransformer,
    DefaultTransformer,
    TransformerRegistry,
)


@pytest.fixture
def registry() -> TransformerRegistry:
    return TransformerRegistry({object: DefaultTransformer()})


def is_string(obj: Any) -> bool:
    return isinstance(obj, str)


def test_conditional_transformer_repr() -> None:
    assert repr(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string)
    ).startswith("ConditionalTransformer(")


def test_conditional_transformer_str() -> None:
    assert str(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string)
    ).startswith("ConditionalTransformer(")


def test_conditional_transformer_transform_condition_true(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string).transform(
            "abc", func=str.upper, registry=registry
        ),
        "ABC",
    )


def test_conditional_transformer_transform_condition_false(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string).transform(
            1, func=str.upper, registry=registry
        ),
        1,
    )
