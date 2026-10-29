from __future__ import annotations

from collections import UserString, defaultdict, deque
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pytest

from coola.equality import objects_are_equal
from coola.recursive import (
    DataclassTransformer,
    DefaultTransformer,
    SequenceTransformer,
    SetTransformer,
    TransformerRegistry,
    get_default_registry,
    interface as recursive_interface,
    recursive_apply,
    register_transformers,
)
from coola.registry import AlreadyRegisteredError
from tests.helpers import CONTAINER_SAMPLES, LEAF_SAMPLES
from tests.unit.recursive.helpers import expand_not_none


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    recursive_interface._default_registry.reset()
    yield
    recursive_interface._default_registry.reset()


class CustomList(list):
    r"""Create a custom class that inherits from list."""


#####################################
#     Tests for recursive_apply     #
#####################################


def test_recursive_apply_with_simple_value() -> None:
    assert recursive_apply(5, lambda x: x * 2) == 10


def test_recursive_apply_with_string() -> None:
    assert recursive_apply("hello", str.upper) == "HELLO"


def test_recursive_apply_with_list() -> None:
    assert objects_are_equal(recursive_apply([1, 2, 3], lambda x: x * 2), [2, 4, 6])


def test_recursive_apply_with_nested_list() -> None:
    assert objects_are_equal(recursive_apply([1, [2, 3], 4], lambda x: x * 2), [2, [4, 6], 8])


def test_recursive_apply_with_dict() -> None:
    assert objects_are_equal(
        recursive_apply({"a": 1, "b": 2}, lambda x: x * 10), {"a": 10, "b": 20}
    )


def test_recursive_apply_with_nested_dict() -> None:
    assert objects_are_equal(
        recursive_apply({"a": 1, "b": {"c": 2}}, lambda x: x * 10), {"a": 10, "b": {"c": 20}}
    )


def test_recursive_apply_with_mixed_nesting() -> None:
    assert objects_are_equal(
        recursive_apply([1, [2, 3], {"x": 4}], lambda x: x * 2), [2, [4, 6], {"x": 8}]
    )


def test_recursive_apply_converts_types() -> None:
    assert objects_are_equal(recursive_apply({"a": 1, "b": "abc"}, str), {"a": "1", "b": "abc"})


def test_recursive_apply_with_custom_registry() -> None:
    assert objects_are_equal(
        recursive_apply(
            [1, 2, 3], lambda x: x * 2, registry=TransformerRegistry({object: DefaultTransformer()})
        ),
        [1, 2, 3, 1, 2, 3],
    )


def test_recursive_apply_with_tuple() -> None:
    assert objects_are_equal(recursive_apply((1, 2, 3), lambda x: x * 2), (2, 4, 6))


def test_recursive_apply_with_set() -> None:
    assert objects_are_equal(recursive_apply({1, 2, 3}, lambda x: x * 2), {2, 4, 6})


###########################################
#     Tests for register_transformers     #
###########################################


def test_register_transformers_calls_registry() -> None:
    register_transformers({CustomList: SequenceTransformer()})
    assert get_default_registry().has_transformer(CustomList)


def test_register_transformers_with_exist_ok_true() -> None:
    register_transformers({CustomList: SetTransformer()})
    register_transformers({CustomList: SequenceTransformer()}, exist_ok=True)


def test_register_transformers_with_exist_ok_false() -> None:
    register_transformers({CustomList: SetTransformer()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        register_transformers({CustomList: SequenceTransformer()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a TransformerRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, TransformerRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


def test_get_default_registry_scalar_types_int() -> None:
    """Test that scalar types are registered with DefaultTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(int)


def test_get_default_registry_scalar_types_float() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(float)


def test_get_default_registry_scalar_types_complex() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(complex)


def test_get_default_registry_scalar_types_bool() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(bool)


def test_get_default_registry_scalar_types_str() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(str)


def test_get_default_registry_sequences_list() -> None:
    """Test that sequence types are registered with
    SequenceTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(list)


def test_get_default_registry_sequences_tuple() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(tuple)


def test_get_default_registry_sequences_deque() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(deque)


def test_get_default_registry_sequences_sequence() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(Sequence)


def test_get_default_registry_sets_set() -> None:
    """Test that set types are registered with SetTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(set)


def test_get_default_registry_sets_frozenset() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(frozenset)


def test_register_default_transformers_registers_mappings_dict() -> None:
    """Test that mapping types are registered with
    MappingTransformer."""
    registry = get_default_registry()
    assert registry.has_transformer(dict)


def test_register_default_transformers_registers_mappings_mapping() -> None:
    registry = get_default_registry()
    assert registry.has_transformer(Mapping)


def test_register_default_transformers_registers_object() -> None:
    """Test that object type is registered as catch-all."""
    registry = get_default_registry()
    assert registry.has_transformer(object)


def test_default_registry_can_transform_list() -> None:
    """Test that default registry can transform a list."""
    registry = get_default_registry()
    assert registry.transform([1, 2, 3], str) == ["1", "2", "3"]


def test_default_registry_can_transform_dict() -> None:
    """Test that default registry can transform a dict."""
    registry = get_default_registry()
    assert registry.transform({"a": 1, "b": 2}, lambda x: x * 10) == {"a": 10, "b": 20}


def test_get_default_registry_singleton_does_not_have_the_custom_type() -> None:
    """Test that the registry does not have the custom type."""
    assert not get_default_registry().has_transformer(CustomList)


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist."""
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceTransformer())
    assert registry1.has_transformer(CustomList)


def test_get_default_registry_singleton_returns_the_same_registry() -> None:
    """Test that the registry is the same across calls."""
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceTransformer())
    assert registry1 is get_default_registry()


def test_get_default_registry_singleton_persists_modifications_across_calls() -> None:
    """Test that modifications to the registry persist across calls."""
    get_default_registry().register(CustomList, SequenceTransformer())
    assert get_default_registry().has_transformer(CustomList)


def test_recursive_apply_with_deque_values() -> None:
    result = recursive_apply(
        deque([1, [2]], maxlen=5), lambda x: x + 1 if isinstance(x, int) else x
    )
    assert result == deque([2, [3]])


def test_recursive_apply_with_deque_maxlen() -> None:
    result = recursive_apply(
        deque([1, [2]], maxlen=5), lambda x: x + 1 if isinstance(x, int) else x
    )
    assert result.maxlen == 5


def test_recursive_apply_with_defaultdict_type() -> None:
    result = recursive_apply(defaultdict(list, {"a": [1]}), lambda x: x + 1)
    assert type(result) is defaultdict


def test_recursive_apply_with_defaultdict_default_factory() -> None:
    result = recursive_apply(defaultdict(list, {"a": [1]}), lambda x: x + 1)
    assert result.default_factory is list


def test_recursive_apply_with_defaultdict_values() -> None:
    result = recursive_apply(defaultdict(list, {"a": [1]}), lambda x: x + 1)
    assert result == {"a": [2]}


def test_recursive_apply_with_mapping_proxy_type() -> None:
    result = recursive_apply(MappingProxyType({"a": 1, "b": [2]}), lambda x: x + 1)
    assert isinstance(result, MappingProxyType)


def test_recursive_apply_with_mapping_proxy_values() -> None:
    result = recursive_apply(MappingProxyType({"a": 1, "b": [2]}), lambda x: x + 1)
    assert result == {"a": 2, "b": [3]}


def test_recursive_apply_with_user_string() -> None:
    assert recursive_apply([UserString("abc")], lambda x: x) == ["abc"]


def test_recursive_apply_keyword_registry() -> None:
    assert recursive_apply([1], str, registry=get_default_registry()) == ["1"]


def test_recursive_apply_positional_registry_raises() -> None:
    with pytest.raises(TypeError, match="takes 2 positional arguments but 3 were given"):
        recursive_apply([1], str, get_default_registry())  # ty: ignore[too-many-positional-arguments]


@dataclass
class Config:
    lr: Any
    layers: list


@pytest.fixture
def local_registry() -> TransformerRegistry:
    return TransformerRegistry({object: DefaultTransformer(), list: SequenceTransformer()})


def test_recursive_apply_enters_the_dataclass() -> None:
    assert recursive_apply(Config(0.1, [1, 2]), str) == Config("0.1", ["1", "2"])


def test_recursive_apply_dataclass_in_a_dict() -> None:
    assert recursive_apply({"config": Config(1, [2])}, str) == {"config": Config("1", ["2"])}


def test_recursive_apply_dataclass_registered_as_a_leaf(
    local_registry: TransformerRegistry,
) -> None:
    local_registry.register(Config, DefaultTransformer())
    assert recursive_apply(Config(0.1, [1]), type, registry=local_registry) is Config


def test_default_registry_dataclass_uses_the_fallback() -> None:
    assert isinstance(get_default_registry().find_transformer(Config), DataclassTransformer)


def test_register_transformers_exist_ok_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        register_transformers({CustomList: SequenceTransformer()}, True)  # ty: ignore[too-many-positional-arguments]


##########################################
#     Tests for the default registry     #
##########################################


@pytest.mark.parametrize("data", [pytest.param(object(), id="object"), *LEAF_SAMPLES])
def test_default_registry_leaf_has_no_expansion(data: Any) -> None:
    assert get_default_registry().find_transformer(type(data)).expand(data) is None


@pytest.mark.parametrize("data", CONTAINER_SAMPLES)
def test_default_registry_rebuild_of_the_unchanged_children_is_equal(data: Any) -> None:
    children, rebuild = expand_not_none(get_default_registry().find_transformer(type(data)), data)
    assert rebuild(list(children)) == data
