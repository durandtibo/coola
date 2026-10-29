from __future__ import annotations

import copy
import multiprocessing
from typing import Any

import pytest

from coola.equality.tester import get_default_registry as get_equality_registry
from coola.hashing import StrHasher, get_default_registry as get_hasher_registry
from coola.iterator import get_default_registry as get_iterator_registry
from coola.random import get_default_registry as get_random_registry
from coola.recursive import get_default_registry as get_transformer_registry
from coola.registry import Registry
from coola.summary import get_default_registry as get_summarizer_registry

TYPE_REGISTRIES = [
    pytest.param(get_equality_registry, id="equality"),
    pytest.param(get_hasher_registry, id="hashing"),
    pytest.param(get_iterator_registry, id="iterator"),
    pytest.param(get_summarizer_registry, id="summary"),
    pytest.param(get_transformer_registry, id="recursive"),
]
DEFAULT_REGISTRIES = [*TYPE_REGISTRIES, pytest.param(get_random_registry, id="random")]


def count_entries(registry: Registry) -> int:
    return len(registry)


def hash_in_worker(registry: Any) -> str:
    return registry.hash("abc", length=8)


###############################################
#     Tests for the default registries        #
###############################################


@pytest.mark.parametrize("get_registry", DEFAULT_REGISTRIES)
def test_default_registry_deepcopy_has_the_same_keys(get_registry: Any) -> None:
    registry = get_registry()
    assert list(copy.deepcopy(registry).keys()) == list(registry.keys())


@pytest.mark.parametrize("get_registry", TYPE_REGISTRIES)
def test_default_registry_deepcopy_has_the_same_fallbacks(get_registry: Any) -> None:
    registry = get_registry()
    assert len(copy.deepcopy(registry).fallbacks()) == len(registry.fallbacks())


@pytest.mark.parametrize("get_registry", DEFAULT_REGISTRIES)
def test_default_registry_deepcopy_is_independent(get_registry: Any) -> None:
    registry = get_registry()
    clone = copy.deepcopy(registry)
    clone.clear()
    assert len(registry) > 0


#############################################
#     Tests for a spawned worker            #
#############################################


def test_registry_is_sent_to_a_spawned_worker() -> None:
    """Test that a registry survives the pickling of a ``spawn``
    worker."""
    registry = Registry[str, int]({"a": 1, "b": 2})
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(count_entries, (registry,)) == 2


def test_default_hasher_registry_is_sent_to_a_spawned_worker() -> None:
    """Test that the default hasher registry survives the pickling of a
    ``spawn`` worker and hashes like the parent."""
    registry = get_hasher_registry().copy()
    registry.register(str, StrHasher(), exist_ok=True)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(hash_in_worker, (registry,)) == registry.hash("abc", length=8)
