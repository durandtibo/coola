from __future__ import annotations

import pytest

import coola.recursive
from coola.recursive import (
    rebuild as rebuild_module,
    rebuild_mapping,
    rebuild_sequence,
    rebuild_set,
)


@pytest.mark.parametrize("name", ["rebuild_mapping", "rebuild_sequence", "rebuild_set"])
def test_rebuild_function_in_all(name: str) -> None:
    assert name in coola.recursive.__all__


def test_rebuild_mapping_is_the_submodule_function() -> None:
    assert rebuild_mapping is rebuild_module.rebuild_mapping


def test_rebuild_sequence_is_the_submodule_function() -> None:
    assert rebuild_sequence is rebuild_module.rebuild_sequence


def test_rebuild_set_is_the_submodule_function() -> None:
    assert rebuild_set is rebuild_module.rebuild_set
