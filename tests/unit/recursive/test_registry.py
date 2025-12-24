from __future__ import annotations

from typing import Any

import pytest

from coola import objects_are_equal
from coola.recursive import (
    BaseTransformer,
    DefaultTransformer,
    MappingTransformer,
    SequenceTransformer,
    SetTransformer,
    TransformerRegistry,
)


class CustomList(list):
    r"""Create a custom class that inherits from list."""


#########################################
#     Tests for TransformerRegistry     #
#########################################


def test_transformer_registry_init_empty() -> None:
    registry = TransformerRegistry()
    assert registry._registry == {}
    assert isinstance(registry._default_transformer, DefaultTransformer)


def test_transformer_registry_init_with_registry() -> None:
    transformer = SequenceTransformer()
    initial_registry: dict[type, BaseTransformer[Any]] = {list: transformer}
    registry = TransformerRegistry(initial_registry)

    assert list in registry._registry
    assert registry._registry[list] is transformer
    # Verify it's a copy
    initial_registry[tuple] = SetTransformer()
    assert tuple not in registry._registry


def test_transformer_registry_repr() -> None:
    assert repr(TransformerRegistry()).startswith("TransformerRegistry(")


def test_transformer_registry_str() -> None:
    assert str(TransformerRegistry()).startswith("TransformerRegistry(")


def test_transformer_registry_register_new_type() -> None:
    registry = TransformerRegistry()
    transformer = SequenceTransformer()
    registry.register(list, transformer)
    assert registry.has_transformer(list)
    assert registry._registry[list] is transformer


def test_transformer_registry_register_existing_type_without_exist_ok() -> None:
    registry = TransformerRegistry()
    transformer1 = SequenceTransformer()
    transformer2 = MappingTransformer()
    registry.register(list, transformer1)
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register(list, transformer2, exist_ok=False)


def test_transformer_registry_register_existing_type_with_exist_ok() -> None:
    registry = TransformerRegistry()
    transformer1 = SequenceTransformer()
    transformer2 = MappingTransformer()

    registry.register(list, transformer1)
    registry.register(list, transformer2, exist_ok=True)

    assert registry._registry[list] is transformer2


def test_transformer_registry_register_many() -> None:
    registry = TransformerRegistry()
    registry.register_many(
        {
            list: SequenceTransformer(),
            dict: MappingTransformer(),
            set: SetTransformer(),
        }
    )
    assert registry.has_transformer(list)
    assert registry.has_transformer(dict)
    assert registry.has_transformer(set)


def test_transformer_registry_register_many_with_existing_type() -> None:
    registry = TransformerRegistry()
    registry.register(list, SequenceTransformer())
    transformers = {
        list: MappingTransformer(),
        dict: MappingTransformer(),
    }
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many(transformers, exist_ok=False)


def test_transformer_registry_register_many_with_exist_ok() -> None:
    registry = TransformerRegistry()
    transformer1 = SequenceTransformer()
    registry.register(list, transformer1)

    transformer2 = MappingTransformer()
    transformers = {
        list: transformer2,
        dict: MappingTransformer(),
    }

    registry.register_many(transformers, exist_ok=True)
    assert registry._registry[list] is transformer2


def test_transformer_registry_register_clears_cache() -> None:
    """Test that registering a new transformer clears the cache."""
    registry = TransformerRegistry()
    # Access find_transformer to potentially populate cache
    assert isinstance(registry.find_transformer(list), DefaultTransformer)
    # Register should clear cache
    transformer = SequenceTransformer()
    registry.register(list, transformer)
    # Verify the new transformer is found
    assert registry.find_transformer(list) is transformer


def test_transformer_registry_has_transformer_true() -> None:
    assert TransformerRegistry({list: SequenceTransformer()}).has_transformer(list)


def test_transformer_registry_has_transformer_false() -> None:
    assert not TransformerRegistry().has_transformer(list)


def test_transformer_registry_find_transformer_direct_match() -> None:
    transformer = SequenceTransformer()
    registry = TransformerRegistry({list: transformer})
    assert registry.find_transformer(list) is transformer


def test_transformer_registry_find_transformer_mro_lookup() -> None:
    transformer = SequenceTransformer()
    registry = TransformerRegistry({list: transformer})
    assert registry.find_transformer(CustomList) is transformer


def test_transformer_registry_find_transformer_default() -> None:
    assert isinstance(TransformerRegistry().find_transformer(str), DefaultTransformer)


def test_transformer_registry_find_transformer_most_specific() -> None:
    base_transformer = SequenceTransformer()
    specific_transformer = MappingTransformer()
    registry = TransformerRegistry({list: base_transformer, CustomList: specific_transformer})

    assert registry.find_transformer(CustomList) is specific_transformer


def test_transformer_registry_transform_with_list() -> None:
    assert objects_are_equal(
        TransformerRegistry({list: SequenceTransformer()}).transform([1, 2, 3], str),
        ["1", "2", "3"],
    )


def test_transformer_registry_transform_with_dict() -> None:
    assert objects_are_equal(
        TransformerRegistry({dict: MappingTransformer()}).transform(
            {"a": 1, "b": 2}, lambda x: x * 2
        ),
        {"a": 2, "b": 4},
    )


def test_transformer_registry_transform_with_nested_structure() -> None:
    registry = TransformerRegistry(
        {
            list: SequenceTransformer(),
            dict: MappingTransformer(),
        }
    )
    assert objects_are_equal(
        registry.transform({"a": [1, 2], "b": [3, 4]}, lambda x: x * 10),
        {"a": [10, 20], "b": [30, 40]},
    )


def test_transformer_registry_transform_with_default_transformer() -> None:
    assert TransformerRegistry().transform(5, lambda x: x * 2) == 10


def test_transformer_registry_transform_with_custom_function() -> None:
    def custom_func(x: int) -> int:
        return x**2

    assert objects_are_equal(
        TransformerRegistry({list: SequenceTransformer()}).transform([2, 3, 4], custom_func),
        [4, 9, 16],
    )


def test_transformer_registry_registry_isolation() -> None:
    transformer1 = SequenceTransformer()
    transformer2 = MappingTransformer()

    registry1 = TransformerRegistry({list: transformer1})
    registry2 = TransformerRegistry({list: transformer2})

    assert registry1._registry[list] is transformer1
    assert registry2._registry[list] is transformer2


def test_transformer_registry_transform_empty_list() -> None:
    assert TransformerRegistry({list: SequenceTransformer()}).transform([], str) == []


def test_transformer_registry_transform_empty_dict() -> None:
    assert TransformerRegistry({dict: MappingTransformer()}).transform({}, str) == {}
