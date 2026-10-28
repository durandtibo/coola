from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pytest

from coola.nested import add_prefix_suffix_to_keys, remove_keys_if
from coola.recursive import BaseTransformer, register_transformers
from tests.integration.helpers import run_threads

NUM_THREADS = 8


@dataclass
class Node:
    name: str
    children: list


class Box:
    def __init__(self, content: Any) -> None:
        self.content = content


class BoxTransformer(BaseTransformer[Box]):
    def expand(self, data: Box) -> Any:
        return [data.content], lambda items: Box(items[0])


def is_a(key: Any) -> bool:
    return key == "a"


def test_remove_keys_if_enters_the_dataclass() -> None:
    node = Node("n", [{"a": 1, "k": 2}])
    assert remove_keys_if({"x": node}, is_a) == {"x": Node("n", [{"k": 2}])}


@pytest.mark.usefixtures("isolated_default_registries")
def test_remove_keys_if_enters_a_registered_container() -> None:
    register_transformers({Box: BoxTransformer()})
    assert remove_keys_if({"b": Box({"a": 1, "c": 2})}, is_a)["b"].content == {"c": 2}


@pytest.mark.usefixtures("isolated_default_registries")
def test_add_prefix_suffix_to_keys_enters_a_registered_container() -> None:
    register_transformers({Box: BoxTransformer()})
    result = add_prefix_suffix_to_keys({"b": Box({"c": 2})}, "x_", recursive=True)
    assert result["x_b"].content == {"x_c": 2}


def test_remove_keys_if_in_threads() -> None:
    """Test that threads can remove keys at the same time with different
    predicates."""
    results: list[dict] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        result = remove_keys_if({"a": 1, "b": {"a": 2, "c": index}}, lambda key: key == "a")
        with lock:
            results.append(result)

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)])

    assert sorted(r["b"]["c"] for r in results) == list(range(NUM_THREADS))
