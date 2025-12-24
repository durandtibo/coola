from __future__ import annotations

from typing import Any

from coola import objects_are_equal
from coola.recursive import TransformerRegistry
from coola.recursive.transformer import ConditionalTransformer, DefaultTransformer


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


def test_conditional_transformer_transform_condition_true() -> None:
    assert objects_are_equal(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string).transform(
            "abc", func=str.upper, registry=TransformerRegistry()
        ),
        "ABC",
    )


def test_conditional_transformer_transform_condition_false() -> None:
    assert objects_are_equal(
        ConditionalTransformer(transformer=DefaultTransformer(), condition=is_string).transform(
            1, func=str.upper, registry=TransformerRegistry()
        ),
        1,
    )
