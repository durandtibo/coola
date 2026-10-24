from __future__ import annotations

from typing import Any

import pytest

from coola.equality import objects_are_equal
from coola.recursive import (
    BaseTransformer,
    DefaultTransformer,
    MappingTransformer,
    SequenceTransformer,
    SetTransformer,
    TransformerRegistry,
    get_default_registry,
)
from coola.registry import AlreadyRegisteredError
from tests.helpers import nest


class CustomList(list):
    r"""Create a custom class that inherits from list."""


#########################################
#     Tests for TransformerRegistry     #
#########################################


def test_transformer_registry_init_empty() -> None:
    registry = TransformerRegistry()
    assert len(registry._state) == 0


def test_transformer_registry_init_with_registry() -> None:
    registry = TransformerRegistry({list: SequenceTransformer()})
    assert list in registry._state


def test_transformer_registry_init_with_registry_keeps_the_transformer() -> None:
    transformer = SequenceTransformer()
    registry = TransformerRegistry({list: transformer})
    assert registry._state[list] is transformer


def test_transformer_registry_init_with_registry_copies_the_registry() -> None:
    initial_registry: dict[type, BaseTransformer[Any]] = {list: SequenceTransformer()}
    registry = TransformerRegistry(initial_registry)
    initial_registry[tuple] = SetTransformer()
    assert tuple not in registry._state


def test_transformer_registry_repr() -> None:
    assert repr(TransformerRegistry()).startswith("TransformerRegistry(")


def test_transformer_registry_str() -> None:
    assert str(TransformerRegistry()).startswith("TransformerRegistry(")


def test_transformer_registry_register_new_type_has_transformer() -> None:
    registry = TransformerRegistry()
    transformer = SequenceTransformer()
    registry.register(list, transformer)
    assert registry.has_transformer(list)


def test_transformer_registry_register_new_type_keeps_the_transformer() -> None:
    registry = TransformerRegistry()
    transformer = SequenceTransformer()
    registry.register(list, transformer)
    assert registry._state[list] is transformer


def test_transformer_registry_register_existing_type_without_exist_ok() -> None:
    registry = TransformerRegistry()
    transformer1 = SequenceTransformer()
    transformer2 = MappingTransformer()
    registry.register(list, transformer1)
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register(list, transformer2, exist_ok=False)


def test_transformer_registry_register_existing_type_with_exist_ok() -> None:
    registry = TransformerRegistry()
    transformer1 = SequenceTransformer()
    transformer2 = MappingTransformer()

    registry.register(list, transformer1)
    registry.register(list, transformer2, exist_ok=True)

    assert registry._state[list] is transformer2


def test_transformer_registry_register_many_list() -> None:
    registry = TransformerRegistry()
    registry.register_many(
        {
            list: SequenceTransformer(),
            dict: MappingTransformer(),
            set: SetTransformer(),
        }
    )
    assert registry.has_transformer(list)


def test_transformer_registry_register_many_dict() -> None:
    registry = TransformerRegistry()
    registry.register_many(
        {
            list: SequenceTransformer(),
            dict: MappingTransformer(),
            set: SetTransformer(),
        }
    )
    assert registry.has_transformer(dict)


def test_transformer_registry_register_many_set() -> None:
    registry = TransformerRegistry()
    registry.register_many(
        {
            list: SequenceTransformer(),
            dict: MappingTransformer(),
            set: SetTransformer(),
        }
    )
    assert registry.has_transformer(set)


def test_transformer_registry_register_many_with_existing_type() -> None:
    registry = TransformerRegistry({list: SequenceTransformer()})
    transformers = {
        list: MappingTransformer(),
        dict: MappingTransformer(),
    }
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register_many(transformers, exist_ok=False)


def test_transformer_registry_register_many_with_exist_ok() -> None:
    registry = TransformerRegistry()
    registry.register(list, SequenceTransformer())

    transformer = MappingTransformer()
    transformers = {
        list: transformer,
        dict: MappingTransformer(),
    }

    registry.register_many(transformers, exist_ok=True)
    assert registry._state[list] is transformer


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


def test_transformer_registry_find_transformer_most_specific() -> None:
    base_transformer = SequenceTransformer()
    specific_transformer = MappingTransformer()
    registry = TransformerRegistry(
        {object: DefaultTransformer(), list: base_transformer, CustomList: specific_transformer}
    )

    assert registry.find_transformer(CustomList) is specific_transformer


def test_transformer_registry_transform_with_list() -> None:
    assert objects_are_equal(
        TransformerRegistry({object: DefaultTransformer(), list: SequenceTransformer()}).transform(
            [1, 2, 3], str
        ),
        ["1", "2", "3"],
    )


def test_transformer_registry_transform_with_dict() -> None:
    assert objects_are_equal(
        TransformerRegistry({object: DefaultTransformer(), dict: MappingTransformer()}).transform(
            {"a": 1, "b": 2}, lambda x: x * 2
        ),
        {"a": 2, "b": 4},
    )


def test_transformer_registry_transform_with_nested_structure() -> None:
    registry = TransformerRegistry(
        {
            object: DefaultTransformer(),
            list: SequenceTransformer(),
            dict: MappingTransformer(),
        }
    )
    assert objects_are_equal(
        registry.transform({"a": [1, 2], "b": [3, 4]}, lambda x: x * 10),
        {"a": [10, 20], "b": [30, 40]},
    )


def test_transformer_registry_transform_with_custom_function() -> None:
    def custom_func(x: int) -> int:
        return x**2

    assert objects_are_equal(
        TransformerRegistry({object: DefaultTransformer(), list: SequenceTransformer()}).transform(
            [2, 3, 4], custom_func
        ),
        [4, 9, 16],
    )


def test_transformer_registry_registry_isolation_first_registry() -> None:
    transformer1 = SequenceTransformer()
    transformer2 = MappingTransformer()

    registry1 = TransformerRegistry({list: transformer1})
    TransformerRegistry({list: transformer2})
    assert registry1._state[list] is transformer1


def test_transformer_registry_registry_isolation_second_registry() -> None:
    transformer1 = SequenceTransformer()
    transformer2 = MappingTransformer()

    TransformerRegistry({list: transformer1})
    registry2 = TransformerRegistry({list: transformer2})
    assert registry2._state[list] is transformer2


def test_transformer_registry_transform_empty_list() -> None:
    assert TransformerRegistry({list: SequenceTransformer()}).transform([], str) == []


def test_transformer_registry_transform_empty_dict() -> None:
    assert TransformerRegistry({dict: MappingTransformer()}).transform({}, str) == {}


@pytest.fixture
def registry() -> TransformerRegistry:
    return get_default_registry()


def test_transformer_registry_transform_list_that_contains_itself(
    registry: TransformerRegistry,
) -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected: an object of type <class 'list'>"):
        registry.transform(data, str)


def test_transformer_registry_transform_dict_that_contains_itself(
    registry: TransformerRegistry,
) -> None:
    data: dict[str, Any] = {}
    data["a"] = data
    with pytest.raises(ValueError, match="Cycle detected: an object of type <class 'dict'>"):
        registry.transform(data, str)


def test_transformer_registry_transform_indirect_cycle(registry: TransformerRegistry) -> None:
    first: list[Any] = []
    second = [first]
    first.append(second)
    with pytest.raises(ValueError, match="Cycle detected"):
        registry.transform(first, str)


def test_transformer_registry_transform_shared_reference_is_not_a_cycle(
    registry: TransformerRegistry,
) -> None:
    shared = [1]
    assert registry.transform([shared, shared], str) == [["1"], ["1"]]


def test_transformer_registry_transform_works_after_a_cycle(registry: TransformerRegistry) -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        registry.transform(data, str)
    assert registry.transform([1, [2]], str) == ["1", ["2"]]


def test_transformer_registry_transform_deep_structure_raises_a_recursion_error(
    registry: TransformerRegistry,
) -> None:
    with pytest.raises(
        RecursionError, match="Python recursion limit was exceeded while transforming"
    ):
        registry.transform(nest(1, 5000), str)


def test_transformer_registry_transform_deep_structure_error_is_raised_once(
    registry: TransformerRegistry,
) -> None:
    with pytest.raises(RecursionError) as exc_info:
        registry.transform(nest(1, 5000), str)
    assert type(exc_info.value.__cause__) is RecursionError


def test_transformer_registry_transform_works_after_a_deep_structure(
    registry: TransformerRegistry,
) -> None:
    with pytest.raises(RecursionError):
        registry.transform(nest(1, 5000), str)
    assert registry.transform([1, [2]], str) == ["1", ["2"]]
