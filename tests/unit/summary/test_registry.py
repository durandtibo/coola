from __future__ import annotations

from typing import Any

import pytest

from coola import objects_are_equal
from coola.summary import (
    BaseSummarizer,
    DefaultSummarizer,
    MappingSummarizer,
    SequenceSummarizer,
    SetSummarizer,
    SummarizerRegistry,
)


class CustomList(list):
    r"""Create a custom class that inherits from list."""


########################################
#     Tests for SummarizerRegistry     #
########################################


def test_summarizer_registry_init_empty() -> None:
    registry = SummarizerRegistry()
    assert len(registry._state) == 0


def test_summarizer_registry_init_with_registry() -> None:
    summarizer = SequenceSummarizer()
    initial_registry: dict[type, BaseSummarizer[Any]] = {list: summarizer}
    registry = SummarizerRegistry(initial_registry)

    assert list in registry._state
    assert registry._state[list] is summarizer
    # Verify it's a copy
    initial_registry[tuple] = SetSummarizer()
    assert tuple not in registry._state


def test_summarizer_registry_repr() -> None:
    assert repr(SummarizerRegistry()).startswith("SummarizerRegistry(")


def test_summarizer_registry_str() -> None:
    assert str(SummarizerRegistry()).startswith("SummarizerRegistry(")


def test_summarizer_registry_register_new_type() -> None:
    registry = SummarizerRegistry()
    summarizer = SequenceSummarizer()
    registry.register(list, summarizer)
    assert registry.has_summarizer(list)
    assert registry._state[list] is summarizer


def test_summarizer_registry_register_existing_type_without_exist_ok() -> None:
    registry = SummarizerRegistry()
    summarizer1 = SequenceSummarizer()
    summarizer2 = MappingSummarizer()
    registry.register(list, summarizer1)
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register(list, summarizer2, exist_ok=False)


def test_summarizer_registry_register_existing_type_with_exist_ok() -> None:
    registry = SummarizerRegistry()
    summarizer1 = SequenceSummarizer()
    summarizer2 = MappingSummarizer()

    registry.register(list, summarizer1)
    registry.register(list, summarizer2, exist_ok=True)

    assert registry._state[list] is summarizer2


def test_summarizer_registry_register_many() -> None:
    registry = SummarizerRegistry()
    registry.register_many(
        {
            list: SequenceSummarizer(),
            dict: MappingSummarizer(),
            set: SetSummarizer(),
        }
    )
    assert registry.has_summarizer(list)
    assert registry.has_summarizer(dict)
    assert registry.has_summarizer(set)


def test_summarizer_registry_register_many_with_existing_type() -> None:
    registry = SummarizerRegistry({list: SequenceSummarizer()})
    summarizers = {
        list: MappingSummarizer(),
        dict: MappingSummarizer(),
    }
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many(summarizers, exist_ok=False)


def test_summarizer_registry_register_many_with_exist_ok() -> None:
    registry = SummarizerRegistry()
    registry.register(list, SequenceSummarizer())

    summarizer = MappingSummarizer()
    summarizers = {
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
    assert SummarizerRegistry({list: SequenceSummarizer()}).summarize([]) == "<class 'list'> []"


def test_summarizer_registry_summarize_empty_dict() -> None:
    assert SummarizerRegistry({dict: MappingSummarizer()}).summarize({}) == "<class 'dict'> {}"


def test_summarizer_registry_registry_isolation() -> None:
    summarizer1 = SequenceSummarizer()
    summarizer2 = MappingSummarizer()

    registry1 = SummarizerRegistry({list: summarizer1})
    registry2 = SummarizerRegistry({list: summarizer2})

    assert registry1._state[list] is summarizer1
    assert registry2._state[list] is summarizer2
