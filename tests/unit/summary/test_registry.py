from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.registry import AlreadyRegisteredError
from coola.summary import (
    DataclassSummarizer,
    DefaultSummarizer,
    MappingSummarizer,
    SequenceSummarizer,
    SetSummarizer,
    SummarizerRegistry,
    SummaryConfig,
)
from tests.helpers import nest

if TYPE_CHECKING:
    from coola.summary.base import Summarizer


class CustomList(list):
    r"""Create a custom class that inherits from list."""


########################################
#     Tests for SummarizerRegistry     #
########################################


def test_summarizer_registry_init_empty() -> None:
    registry = SummarizerRegistry()
    assert len(registry._state) == 0


def test_summarizer_registry_init_with_registry() -> None:
    registry = SummarizerRegistry({list: SequenceSummarizer()})
    assert list in registry._state


def test_summarizer_registry_init_with_registry_keeps_the_summarizer() -> None:
    summarizer = SequenceSummarizer()
    registry = SummarizerRegistry({list: summarizer})
    assert registry._state[list] is summarizer


def test_summarizer_registry_init_with_registry_copies_the_registry() -> None:
    initial_registry: dict[type, Summarizer[Any]] = {list: SequenceSummarizer()}
    registry = SummarizerRegistry(initial_registry)
    initial_registry[tuple] = SetSummarizer()
    assert tuple not in registry._state


def test_summarizer_registry_repr() -> None:
    assert repr(SummarizerRegistry()).startswith("SummarizerRegistry(")


def test_summarizer_registry_str() -> None:
    assert str(SummarizerRegistry()).startswith("SummarizerRegistry(")


def test_summarizer_registry_register_new_type_has_summarizer() -> None:
    registry = SummarizerRegistry()
    summarizer = SequenceSummarizer()
    registry.register(list, summarizer)
    assert registry.has_summarizer(list)


def test_summarizer_registry_register_new_type_keeps_the_summarizer() -> None:
    registry = SummarizerRegistry()
    summarizer = SequenceSummarizer()
    registry.register(list, summarizer)
    assert registry._state[list] is summarizer


def test_summarizer_registry_register_existing_type_without_exist_ok() -> None:
    registry = SummarizerRegistry()
    summarizer1 = SequenceSummarizer()
    summarizer2 = MappingSummarizer()
    registry.register(list, summarizer1)
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register(list, summarizer2, exist_ok=False)


def test_summarizer_registry_register_existing_type_with_exist_ok() -> None:
    registry = SummarizerRegistry()
    summarizer1 = SequenceSummarizer()
    summarizer2 = MappingSummarizer()

    registry.register(list, summarizer1)
    registry.register(list, summarizer2, exist_ok=True)

    assert registry._state[list] is summarizer2


def test_summarizer_registry_register_many_list() -> None:
    registry = SummarizerRegistry()
    registry.register_many(
        {
            list: SequenceSummarizer(),
            dict: MappingSummarizer(),
            set: SetSummarizer(),
        }
    )
    assert registry.has_summarizer(list)


def test_summarizer_registry_register_many_dict() -> None:
    registry = SummarizerRegistry()
    registry.register_many(
        {
            list: SequenceSummarizer(),
            dict: MappingSummarizer(),
            set: SetSummarizer(),
        }
    )
    assert registry.has_summarizer(dict)


def test_summarizer_registry_register_many_set() -> None:
    registry = SummarizerRegistry()
    registry.register_many(
        {
            list: SequenceSummarizer(),
            dict: MappingSummarizer(),
            set: SetSummarizer(),
        }
    )
    assert registry.has_summarizer(set)


def test_summarizer_registry_register_many_with_existing_type() -> None:
    registry = SummarizerRegistry({list: SequenceSummarizer()})
    summarizers: dict[type, Summarizer[Any]] = {
        list: MappingSummarizer(),
        dict: MappingSummarizer(),
    }
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register_many(summarizers, exist_ok=False)


def test_summarizer_registry_register_many_with_exist_ok() -> None:
    registry = SummarizerRegistry()
    registry.register(list, SequenceSummarizer())

    summarizer = MappingSummarizer()
    summarizers: dict[type, Summarizer[Any]] = {
        list: summarizer,
        dict: MappingSummarizer(),
    }

    registry.register_many(summarizers, exist_ok=True)
    assert registry._state[list] is summarizer


def test_summarizer_registry_has_summarizer_true() -> None:
    assert SummarizerRegistry({list: SequenceSummarizer()}).has_summarizer(list)


def test_summarizer_registry_has_summarizer_false() -> None:
    assert not SummarizerRegistry().has_summarizer(list)


def test_summarizer_registry_find_summarizer_direct_match() -> None:
    summarizer = SequenceSummarizer()
    registry = SummarizerRegistry({list: summarizer})
    assert registry.find_summarizer(list) is summarizer


def test_summarizer_registry_find_summarizer_mro_lookup() -> None:
    summarizer = SequenceSummarizer()
    registry = SummarizerRegistry({list: summarizer})
    assert registry.find_summarizer(CustomList) is summarizer


def test_summarizer_registry_find_summarizer_most_specific() -> None:
    base_summarizer = SequenceSummarizer()
    specific_summarizer = MappingSummarizer()
    registry = SummarizerRegistry(
        {object: DefaultSummarizer(), list: base_summarizer, CustomList: specific_summarizer}
    )

    assert registry.find_summarizer(CustomList) is specific_summarizer


def test_summarizer_registry_summarize_with_list() -> None:
    assert objects_are_equal(
        SummarizerRegistry({object: DefaultSummarizer(), list: SequenceSummarizer()}).summarize(
            [1, 2, 3]
        ),
        "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3",
    )


def test_summarizer_registry_summarize_with_dict() -> None:
    assert objects_are_equal(
        SummarizerRegistry({object: DefaultSummarizer(), dict: MappingSummarizer()}).summarize(
            {"a": 1, "b": 2}
        ),
        "<class 'dict'> (length=2)\n  (a): 1\n  (b): 2",
    )


def test_summarizer_registry_summarize_with_nested_structure() -> None:
    registry = SummarizerRegistry(
        {
            object: DefaultSummarizer(),
            list: SequenceSummarizer(),
            dict: MappingSummarizer(),
        }
    )
    assert objects_are_equal(
        registry.summarize({"a": [1, 2], "b": [3, 4]}),
        "<class 'dict'> (length=2)\n  (a): [1, 2]\n  (b): [3, 4]",
    )


def test_summarizer_registry_summarize_empty_list() -> None:
    assert (
        SummarizerRegistry({list: SequenceSummarizer()}).summarize([])
        == "<class 'list'> (length=0)"
    )


def test_summarizer_registry_summarize_empty_dict() -> None:
    assert (
        SummarizerRegistry({dict: MappingSummarizer()}).summarize({}) == "<class 'dict'> (length=0)"
    )


def test_summarizer_registry_registry_isolation_first_registry() -> None:
    summarizer1 = SequenceSummarizer()
    summarizer2 = MappingSummarizer()

    registry1 = SummarizerRegistry({list: summarizer1})
    SummarizerRegistry({list: summarizer2})
    assert registry1._state[list] is summarizer1


def test_summarizer_registry_registry_isolation_second_registry() -> None:
    summarizer1 = SequenceSummarizer()
    summarizer2 = MappingSummarizer()

    SummarizerRegistry({list: summarizer1})
    registry2 = SummarizerRegistry({list: summarizer2})
    assert registry2._state[list] is summarizer2


def test_summarizer_registry_summarize_with_config() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer(), list: SequenceSummarizer()})
    assert registry.summarize([1, 2, 3], SummaryConfig(max_items=1)) == (
        "<class 'list'> (length=3)\n  (0): 1\n  ... (+2 more)"
    )


def test_summarizer_registry_summarize_without_config_uses_the_default_options() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer(), list: SequenceSummarizer()})
    assert registry.summarize([1]) == registry.summarize([1], SummaryConfig())


def test_summarizer_registry_summarize_deep_structure_does_not_raise() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer(), list: SequenceSummarizer()})
    assert registry.summarize(nest(1, 5000), SummaryConfig(max_depth=10**6)).count("\n") == 5000


def test_summarizer_registry_summarize_without_object_summarizer_raises() -> None:
    with pytest.raises(KeyError, match="int"):
        SummarizerRegistry().summarize(1)


def test_summarizer_registry_find_summarizer_dataclass_with_a_fallback() -> None:
    @dataclass
    class Point:
        x: int = 1

    registry = SummarizerRegistry({object: DefaultSummarizer()})
    registry.register_fallback(is_dataclass, DataclassSummarizer())
    assert isinstance(registry.find_summarizer(Point), DataclassSummarizer)


def test_summarizer_registry_find_summarizer_dataclass_with_registered_base() -> None:
    """Regression test: a summarizer shared by ``object`` and a base
    class is used for a dataclass that inherits from the base class."""

    class Base:
        pass

    @dataclass
    class Derived(Base):
        x: int = 1

    shared = DefaultSummarizer()
    registry = SummarizerRegistry({object: shared, Base: shared})
    registry.register_fallback(is_dataclass, DataclassSummarizer())
    assert registry.find_summarizer(Derived) is shared
