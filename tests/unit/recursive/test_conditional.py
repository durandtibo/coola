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
    """Test ConditionalTransformer applies transformation when condition
    is True."""
    assert objects_are_equal(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string).transform(
            "abc", func=str.upper, registry=registry
        ),
        "ABC",
    )


def test_conditional_transformer_transform_condition_false(registry: TransformerRegistry) -> None:
    """Test ConditionalTransformer skips transformation when condition
    is False."""
    assert objects_are_equal(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string).transform(
            1, func=str.upper, registry=registry
        ),
        1,
    )


def test_conditional_transformer_transform_with_none(registry: TransformerRegistry) -> None:
    """Test ConditionalTransformer handles None input correctly."""
    assert objects_are_equal(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string).transform(
            None, func=str.upper, registry=registry
        ),
        None,
    )


def test_conditional_transformer_transform_complex_condition(registry: TransformerRegistry) -> None:
    """Test ConditionalTransformer with complex condition (string length
    > 5)."""

    def is_long_string(obj: Any) -> bool:
        return isinstance(obj, str) and len(obj) > 5

    transformer = ConditionalTransformer(transformer=DefaultTransformer(), condition=is_long_string)
    # Long string - condition True, should transform
    assert objects_are_equal(
        transformer.transform("verylongstring", func=str.upper, registry=registry), "VERYLONGSTRING"
    )
    # Short string - condition False, should not transform
    assert objects_are_equal(
        transformer.transform("short", func=str.upper, registry=registry), "short"
    )
