from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.recursive import (
    BaseTransformer,
    DefaultTransformer,
    IdentityTransformer,
    KeyFilterTransformer,
    SequenceTransformer,
    SetTransformer,
    Transformer,
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


def test_key_filter_transformer_transform_preserves_mapping_type_type(
    registry: TransformerRegistry,
) -> None:
    result = KeyFilterTransformer().transform(
        OrderedDict(key=1, b=2), func=lambda key: key == "key", registry=registry
    )
    assert isinstance(result, OrderedDict)


def test_key_filter_transformer_transform_preserves_mapping_type_values(
    registry: TransformerRegistry,
) -> None:
    result = KeyFilterTransformer().transform(
        OrderedDict(key=1, b=2), func=lambda key: key == "key", registry=registry
    )
    assert result == {"b": 2}


def test_key_filter_transformer_transform_with_exception_in_predicate(
    registry: TransformerRegistry,
) -> None:
    def failing_predicate(key: Any) -> bool:
        msg = f"Test error {key}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match=r"Test error"):
        KeyFilterTransformer().transform({"a": 1}, func=failing_predicate, registry=registry)


###############################################################
#     Tests for the predicate / transform_func decoupling     #
###############################################################


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


def test_key_filter_transformer_transform_default_transform_func_leaves_values_unchanged(
    transforming_registry: TransformerRegistry,
) -> None:
    # Without a transform_func, the kept values are left unchanged: the key
    # predicate is not applied to them (it used to replace them with booleans).
    assert objects_are_equal(
        KeyFilterTransformer().transform(
            {"keep": 1, "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
        ),
        {"keep": 1},
    )


def test_key_filter_transformer_transform_explicit_transform_func_decouples_values(
    transforming_registry: TransformerRegistry,
) -> None:
    # An explicit transform_func decouples value transformation from the
    # key predicate: kept values are left untouched by the identity func.
    assert objects_are_equal(
        KeyFilterTransformer(transform_func=lambda x: x).transform(
            {"keep": 1, "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
        ),
        {"keep": 1},
    )


def test_key_filter_transformer_transform_transform_func_applied_to_values(
    transforming_registry: TransformerRegistry,
) -> None:
    assert objects_are_equal(
        KeyFilterTransformer(transform_func=str).transform(
            {"keep": 1, "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
        ),
        {"keep": "1"},
    )


def test_key_filter_transformer_transform_str_value_uses_transform_func(
    transforming_registry: TransformerRegistry,
) -> None:
    # str values must use transform_func, not func, even though str is a
    # Sequence: otherwise the predicate would be misapplied to string values.
    assert objects_are_equal(
        KeyFilterTransformer(transform_func=str.upper).transform(
            {"keep": "hello", "secret": 2},
            func=lambda key: key == "secret",
            registry=transforming_registry,
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


#########################################
#     Tests for the Transformer API     #
#########################################


def test_key_filter_transformer_repr_default() -> None:
    assert (
        repr(KeyFilterTransformer())
        == "KeyFilterTransformer(key_predicate=None, transform_func=None)"
    )


def test_key_filter_transformer_str_default() -> None:
    assert (
        str(KeyFilterTransformer())
        == "KeyFilterTransformer(key_predicate=None, transform_func=None)"
    )


def test_key_filter_transformer_repr_with_transform_func() -> None:
    assert repr(KeyFilterTransformer(transform_func=str)).startswith(
        "KeyFilterTransformer(key_predicate=None, transform_func="
    )


def test_key_filter_transformer_follows_the_transformer_protocol(
    registry: TransformerRegistry,
) -> None:
    # ``func`` and ``registry`` are the only arguments, like for the other transformers.
    transformer: Transformer[Any] = KeyFilterTransformer()
    assert transformer.transform({"a": 1, "b": 2}, lambda key: key == "a", registry) == {"b": 2}


def test_key_filter_transformer_is_a_base_transformer() -> None:
    assert isinstance(KeyFilterTransformer(), BaseTransformer)


def test_key_filter_transformer_transform_rejects_transform_func_argument(
    registry: TransformerRegistry,
) -> None:
    with pytest.raises(TypeError, match="transform_func"):
        KeyFilterTransformer().transform(  # type: ignore[call-arg]
            {"a": 1}, lambda _: False, registry, transform_func=str
        )


def test_key_filter_transformer_transform_defaultdict_type(registry: TransformerRegistry) -> None:
    result = KeyFilterTransformer().transform(
        defaultdict(int, {"a": 1, "b": 2}), func=lambda key: key == "a", registry=registry
    )
    assert type(result) is defaultdict


def test_key_filter_transformer_transform_defaultdict_default_factory(
    registry: TransformerRegistry,
) -> None:
    result = KeyFilterTransformer().transform(
        defaultdict(int, {"a": 1, "b": 2}), func=lambda key: key == "a", registry=registry
    )
    assert result.default_factory is int


def test_key_filter_transformer_transform_defaultdict_values(registry: TransformerRegistry) -> None:
    result = KeyFilterTransformer().transform(
        defaultdict(int, {"a": 1, "b": 2}), func=lambda key: key == "a", registry=registry
    )
    assert result == {"b": 2}


##########################################
#     Tests with the key predicate      #
##########################################


def is_secret(key: Any) -> bool:
    return key == "secret"


@pytest.fixture
def predicate_registry() -> TransformerRegistry:
    r"""A registry whose leaf transformer applies the function."""
    key_filter = KeyFilterTransformer(key_predicate=is_secret)
    return TransformerRegistry(
        {
            object: DefaultTransformer(),
            list: SequenceTransformer(),
            tuple: SequenceTransformer(),
            set: SetTransformer(),
            dict: key_filter,
            OrderedDict: key_filter,
        }
    )


def test_key_filter_transformer_key_predicate_drops_the_keys(
    predicate_registry: TransformerRegistry,
) -> None:
    assert objects_are_equal(
        predicate_registry.transform({"keep": 1, "secret": 2}, lambda x: x), {"keep": 1}
    )


def test_key_filter_transformer_key_predicate_func_is_applied_to_the_values(
    predicate_registry: TransformerRegistry,
) -> None:
    assert objects_are_equal(
        predicate_registry.transform({"keep": 1, "secret": 2}, str), {"keep": "1"}
    )


def test_key_filter_transformer_key_predicate_func_is_applied_to_nested_values(
    predicate_registry: TransformerRegistry,
) -> None:
    data = {"a": {"secret": 1, "b": [1, {"secret": 2, "c": 3}]}, "secret": 4}
    assert objects_are_equal(
        predicate_registry.transform(data, lambda x: x * 10), {"a": {"b": [10, {"c": 30}]}}
    )


def test_key_filter_transformer_key_predicate_is_not_applied_to_the_values(
    predicate_registry: TransformerRegistry,
) -> None:
    # The values are not keys: a value equal to "secret" is kept.
    assert objects_are_equal(
        predicate_registry.transform({"keep": "secret", "x": ["secret"]}, lambda x: x),
        {"keep": "secret", "x": ["secret"]},
    )


def test_key_filter_transformer_key_predicate_keeps_the_type_type(
    predicate_registry: TransformerRegistry,
) -> None:
    result = predicate_registry.transform(OrderedDict(a=1, secret=2), lambda x: x)
    assert type(result) is OrderedDict


def test_key_filter_transformer_key_predicate_keeps_the_type_values(
    predicate_registry: TransformerRegistry,
) -> None:
    result = predicate_registry.transform(OrderedDict(a=1, secret=2), lambda x: x)
    assert result == OrderedDict(a=1)


def test_key_filter_transformer_key_predicate_ignores_the_predicate_forwarded_as_func(
    predicate_registry: TransformerRegistry,
) -> None:
    # In the legacy mode, ``func`` is the predicate. With a ``key_predicate``, ``func`` is
    # the function of the values, so a predicate given as ``func`` is applied to the values.
    result = predicate_registry.transform({"keep": 1, "secret": 2}, lambda key: key == "keep")
    assert objects_are_equal(result, {"keep": False})


def test_key_filter_transformer_key_predicate_and_transform_func_raises() -> None:
    with pytest.raises(ValueError, match="`transform_func` is only used without `key_predicate`"):
        KeyFilterTransformer(key_predicate=is_secret, transform_func=str)


def test_key_filter_transformer_key_predicate_repr() -> None:
    assert repr(KeyFilterTransformer(key_predicate=is_secret)).startswith(
        "KeyFilterTransformer(key_predicate=<function is_secret"
    )


def test_key_filter_transformer_key_predicate_follows_the_transformer_protocol() -> None:
    transformer: Transformer[Any] = KeyFilterTransformer(key_predicate=is_secret)
    assert isinstance(transformer, KeyFilterTransformer)
