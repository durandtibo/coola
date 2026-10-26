from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any

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


def is_key(key: Any) -> bool:
    return isinstance(key, str) and "key" in key


def is_a(key: Any) -> bool:
    return key == "a"


def is_key_literal(key: Any) -> bool:
    return key == "key"


def always(key: Any) -> bool:  # noqa: ARG001
    return True


def never(key: Any) -> bool:  # noqa: ARG001
    return False


@pytest.fixture
def registry() -> TransformerRegistry:
    r"""A registry whose leaves are kept as they are."""
    identity = IdentityTransformer()
    key_filter = KeyFilterTransformer(is_key)
    return TransformerRegistry(
        {
            object: identity,
            str: identity,
            list: SequenceTransformer(),
            tuple: SequenceTransformer(),
            set: SetTransformer(),
            frozenset: SetTransformer(),
            dict: key_filter,
            OrderedDict: key_filter,
            defaultdict: key_filter,
        }
    )


def test_key_filter_transformer_drop_one_key(registry: TransformerRegistry) -> None:
    assert objects_are_equal(registry.transform({"key": 1, "b": 2}, str), {"b": 2})


def test_key_filter_transformer_drop_nothing(registry: TransformerRegistry) -> None:
    assert objects_are_equal(registry.transform({"a": 1, "b": 2}, str), {"a": 1, "b": 2})


def test_key_filter_transformer_drop_everything(registry: TransformerRegistry) -> None:
    assert objects_are_equal(registry.transform({"key": 1, "key2": 2}, str), {})


def test_key_filter_transformer_empty_mapping(registry: TransformerRegistry) -> None:
    assert objects_are_equal(registry.transform({}, str), {})


def test_key_filter_transformer_non_string_key_kept(registry: TransformerRegistry) -> None:
    assert objects_are_equal(registry.transform({"key": 1, "abc": 2, 1: 3}, str), {"abc": 2, 1: 3})


def test_key_filter_transformer_nested_mapping(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({"a": {"key": 1, "b": 2}, "key": {"b": 3}}, str), {"a": {"b": 2}}
    )


def test_key_filter_transformer_nested_list(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({"list": [{"key": 1, "b": 2}, {"key": 3}], "b": 4}, str),
        {"list": [{"b": 2}, {}], "b": 4},
    )


def test_key_filter_transformer_nested_tuple(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({"tuple": ({"key": 1, "b": 2},)}, str), {"tuple": ({"b": 2},)}
    )


def test_key_filter_transformer_nested_set(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({"set": {1, 2, 3}, "key": 4}, str), {"set": {1, 2, 3}}
    )


def test_key_filter_transformer_preserves_mapping_type(registry: TransformerRegistry) -> None:
    assert type(registry.transform(OrderedDict(key=1, b=2), str)) is OrderedDict


def test_key_filter_transformer_preserves_mapping_values(registry: TransformerRegistry) -> None:
    assert registry.transform(OrderedDict(key=1, b=2), str) == {"b": 2}


def test_key_filter_transformer_defaultdict_type(registry: TransformerRegistry) -> None:
    assert type(registry.transform(defaultdict(int, {"a": 1, "key": 2}), str)) is defaultdict


def test_key_filter_transformer_defaultdict_default_factory(registry: TransformerRegistry) -> None:
    result = registry.transform(defaultdict(int, {"a": 1, "key": 2}), str)
    assert result.default_factory is int


def test_key_filter_transformer_defaultdict_values(registry: TransformerRegistry) -> None:
    assert registry.transform(defaultdict(int, {"a": 1, "key": 2}), str) == {"a": 1}


def test_key_filter_transformer_predicate_called_once_per_key() -> None:
    keys = []

    def predicate(key: Any) -> bool:
        keys.append(key)
        return False

    registry = TransformerRegistry(
        {object: DefaultTransformer(), dict: KeyFilterTransformer(predicate)}
    )
    registry.transform({"a": {"b": 1}, "c": 2}, str)
    assert keys == ["a", "c", "b"]


def test_key_filter_transformer_exception_in_predicate() -> None:
    def failing_predicate(key: Any) -> bool:
        msg = f"Test error {key}"
        raise ValueError(msg)

    registry = TransformerRegistry(
        {object: DefaultTransformer(), dict: KeyFilterTransformer(failing_predicate)}
    )
    with pytest.raises(ValueError, match=r"Test error"):
        registry.transform({"a": 1}, str)


def test_key_filter_transformer_is_a_base_transformer() -> None:
    assert isinstance(KeyFilterTransformer(is_a), BaseTransformer)


def test_key_filter_transformer_follows_the_transformer_protocol() -> None:
    transformer: Transformer[Any] = KeyFilterTransformer(is_a)
    children, rebuild = transformer.expand({"a": 1, "b": 2})
    assert rebuild(children) == {"b": 2}


def test_key_filter_transformer_key_predicate_is_required() -> None:
    with pytest.raises(TypeError, match="key_predicate"):
        KeyFilterTransformer()  # type: ignore[call-arg]


def test_key_filter_transformer_repr() -> None:
    assert repr(KeyFilterTransformer(is_a)).startswith(
        "KeyFilterTransformer(key_predicate=<function is_a"
    )


def test_key_filter_transformer_str() -> None:
    assert str(KeyFilterTransformer(is_a)).startswith(
        "KeyFilterTransformer(key_predicate=<function is_a"
    )


def test_key_filter_transformer_drop_nothing_never() -> None:
    registry = TransformerRegistry(
        {object: DefaultTransformer(), dict: KeyFilterTransformer(never)}
    )
    assert registry.transform({"a": 1}, str) == {"a": "1"}


def test_key_filter_transformer_drop_everything_always() -> None:
    registry = TransformerRegistry(
        {object: DefaultTransformer(), dict: KeyFilterTransformer(always)}
    )
    assert registry.transform({"a": 1}, str) == {}


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


def test_key_filter_transformer_key_predicate_is_not_applied_as_func(
    predicate_registry: TransformerRegistry,
) -> None:
    # ``func`` is the function of the values, so a predicate given as ``func`` is applied to them.
    result = predicate_registry.transform({"keep": 1, "secret": 2}, lambda key: key == "keep")
    assert objects_are_equal(result, {"keep": False})


def test_key_filter_transformer_expand_children() -> None:
    children, _ = KeyFilterTransformer(is_a).expand({"a": 1, "b": 2, "c": 3})
    assert children == [2, 3]


def test_key_filter_transformer_expand_rebuild() -> None:
    _, rebuild = KeyFilterTransformer(is_a).expand({"a": 1, "b": 2, "c": 3})
    assert rebuild(["x", "y"]) == {"b": "x", "c": "y"}


def test_key_filter_transformer_expand_empty() -> None:
    children, rebuild = KeyFilterTransformer(is_a).expand({})
    assert (children, rebuild([])) == ([], {})


def test_key_filter_transformer_expand_does_not_modify_the_input() -> None:
    data = {"a": 1, "b": 2}
    KeyFilterTransformer(is_a).expand(data)
    assert data == {"a": 1, "b": 2}
