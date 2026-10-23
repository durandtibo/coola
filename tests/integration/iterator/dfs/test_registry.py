from __future__ import annotations

from coola.iterator.dfs import DefaultIterator, IterableIterator, IteratorRegistry

#######################################
#     Tests for IteratorRegistry      #
#######################################


def test_registry_iterate_very_wide_structure() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    data = [[i] for i in range(20_000)]
    assert sum(registry.iterate(data)) == 199990000
