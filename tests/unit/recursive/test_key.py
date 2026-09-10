from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.recursive import (
    DefaultTransformer,
    IdentityTransformer,
    KeyFilterTransformer,
    SequenceTransformer,
    SetTransformer,
    TransformerRegistry,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    identity = IdentityTransformer()
    key_filter = KeyFilterTransformer()
    return TransformerRegistry(
        {
            object: identity,
            str: identity,
            list: SequenceTransformer(),
            tuple: SequenceTransformer(),
            set: SetTransformer(),
            frozenset: SetTransformer(),
            dict: key_filter,
        }
    )


@pytest.mark.parametrize(
    ("data", "predicate", "expected"),
    [
        pytest.param({"a": 1, "b": 2}, lambda key: key == "a", {"b": 2}, id="drop_one_key"),
        pytest.param(
            {"a": 1, "b": 2},
            lambda key: False,  # noqa: ARG005
            {"a": 1, "b": 2},
            id="drop_nothing",
        ),
        pytest.param({"a": 1, "b": 2}, lambda key: True, {}, id="drop_everything"),  # noqa: ARG005
        pytest.param({}, lambda key: key == "a", {}, id="empty_mapping"),
        pytest.param(
            {"key": 1, "abc": 2, 1: 3},
            lambda key: isinstance(key, str) and "key" in key,
            {"abc": 2, 1: 3},
            id="non_string_key_kept",
        ),
    ],
)
def test_key_filter_transformer_transform_parametrized(
    data: dict, predicate: Callable, expected: dict, registry: TransformerRegistry
) -> None:
    assert objects_are_equal(
        KeyFilterTransformer().transform(data, func=predicate, registry=registry), expected
    )


def test_key_filter_transformer_transform_nested_mapping(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"a": {"key": 1, "b": 2}, "key": {"b": 3}},
            func=lambda key: isinstance(key, str) and "key" in key,
            registry=registry,
        ),
        {"a": {"b": 2}},
    )


def test_key_filter_transformer_transform_nested_list(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"list": [{"key": 1, "b": 2}, {"key": 3}], "b": 4},
            func=lambda key: isinstance(key, str) and "key" in key,
            registry=registry,
        ),
        {"list": [{"b": 2}, {}], "b": 4},
    )


def test_key_filter_transformer_transform_nested_tuple(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"tuple": ({"key": 1, "b": 2},)},
            func=lambda key: isinstance(key, str) and "key" in key,
            registry=registry,
        ),
        {"tuple": ({"b": 2},)},
    )


def test_key_filter_transformer_transform_nested_set(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"set": {1, 2, 3}, "key": 4},
            func=lambda key: isinstance(key, str) and "key" in key,
            registry=registry,
        ),
        {"set": {1, 2, 3}},
    )


def test_key_filter_transformer_transform_preserves_mapping_type(
    registry: TransformerRegistry,
) -> None:
    result = KeyFilterTransformer().transform(
        OrderedDict(key=1, b=2), func=lambda key: key == "key", registry=registry
    )
    assert isinstance(result, OrderedDict)
    assert result == {"b": 2}


def test_key_filter_transformer_transform_with_exception_in_predicate(
    registry: TransformerRegistry,
) -> None:
    def failing_predicate(key: Any) -> bool:
        msg = f"Test error {key}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match=r"Test error"):
        KeyFilterTransformer().transform({"a": 1}, func=failing_predicate, registry=registry)


##############################################################
#     Tests for the predicate / transform_func decoupling    #
##############################################################


@pytest.fixture
def transforming_registry() -> TransformerRegistry:
    # A registry where non-container values are actually transformed by
    # the function they receive, unlike the ``registry`` fixture above
    # where non-container values are left untouched (``IdentityTransformer``).
    return TransformerRegistry(
        {
            object: DefaultTransformer(),
            list: SequenceTransformer(),
            tuple: SequenceTransformer(),
            set: SetTransformer(),
            frozenset: SetTransformer(),
            dict: KeyFilterTransformer(),
        }
    )


def test_key_filter_transformer_transform_default_transform_func_reuses_predicate(
    transforming_registry: TransformerRegistry,
) -> None:
    # Without an explicit transform_func, kept values are run through the
    # key predicate too, reproducing the surprising legacy behavior.
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"keep": 1, "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
        ),
        {"keep": False},
    )


def test_key_filter_transformer_transform_explicit_transform_func_decouples_values(
    transforming_registry: TransformerRegistry,
) -> None:
    # An explicit transform_func decouples value transformation from the
    # key predicate: kept values are left untouched by the identity func.
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"keep": 1, "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
            transform_func=lambda x: x,
        ),
        {"keep": 1},
    )


def test_key_filter_transformer_transform_transform_func_applied_to_values(
    transforming_registry: TransformerRegistry,
) -> None:
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"keep": 1, "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
            transform_func=str,
        ),
        {"keep": "1"},
    )


def test_key_filter_transformer_transform_str_value_uses_transform_func(
    transforming_registry: TransformerRegistry,
) -> None:
    # str values must use transform_func, not func, even though str is a
    # Sequence: otherwise the predicate would be misapplied to string values.
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"keep": "hello", "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
            transform_func=str.upper,
        ),
        {"keep": "HELLO"},
    )


def test_key_filter_transformer_transform_default_transform_func_nested_mapping_still_filtered(
    registry: TransformerRegistry,
) -> None:
    # With the default (no explicit transform_func), the predicate keeps
    # filtering nested mappings at every level, exactly as before this fix.
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"a": {"key": 1, "b": 2}, "key": {"b": 3}},
            func=lambda key: isinstance(key, str) and "key" in key,
            registry=registry,
        ),
        {"a": {"b": 2}},
    )
