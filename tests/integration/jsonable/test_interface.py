from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from coola.jsonable import to_jsonable
from tests.integration.helpers import run_threads

NUM_THREADS = 8

SCRIPT = """
from dataclasses import dataclass
from coola.jsonable import to_jsonable

@dataclass
class P:
    tags: set
    corner: tuple

print(to_jsonable({"p": P({"d", "a", "b", "c", "e"}, (1, 2)), "s": {"x", "y", 1, 2}}))
"""


@dataclass
class Node:
    value: Any
    children: list[Any]


def run(seed: str) -> str:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT],
        env={"PYTHONHASHSEED": seed, "PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.parametrize("seed", ["0", "1", "2", "3", "42"])
def test_to_jsonable_is_the_same_for_every_hash_seed(seed: str) -> None:
    """Test that the output does not depend on ``PYTHONHASHSEED``."""
    assert run(seed) == (
        "{'p': {'tags': ['a', 'b', 'c', 'd', 'e'], 'corner': [1, 2]}, 's': [1, 2, 'x', 'y']}\n"
    )


def test_to_jsonable_concurrent_calls_do_not_share_state() -> None:
    """Test that concurrent calls of ``to_jsonable`` on the default
    registry return the right result for their own data."""
    results: list[Any] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        for _ in range(50):
            result = to_jsonable({"a": [index, (index, {index})], "b": index})
            with lock:
                results.append(result)

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)])
    assert len(results) == NUM_THREADS * 50
    assert {r["b"] for r in results} == set(range(NUM_THREADS))
    assert all(r["a"] == [r["b"], [r["b"], [r["b"]]]] for r in results)


def test_to_jsonable_concurrent_calls_detect_cycles() -> None:
    """Test that a cycle in one thread does not affect the calls of the
    other threads."""
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        data: list[Any] = [index]
        data.append(data)
        try:
            to_jsonable(data if index % 2 else [index])
        except ValueError as exc:
            with lock:
                errors.append(str(exc))

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)])
    assert len(errors) == NUM_THREADS // 2


def test_to_jsonable_deep_nesting_does_not_raise() -> None:
    """Test that a list nested 100,000 levels is converted."""
    data: list[Any] = []
    for _ in range(100_000):
        data = [data]
    result = to_jsonable(data)
    depth = 0
    while result:
        result = result[0]
        depth += 1
    assert depth == 100_000


@pytest.mark.parametrize("kind", ["list", "dict", "dataclass"])
def test_to_jsonable_cycle_raises(kind: str) -> None:
    """Test that a cycle through a list, a dict and a dataclass raises a
    ``ValueError``."""
    node = Node(1, [])
    mapping: dict[str, Any] = {}
    node.children.append(mapping)
    mapping["node"] = node
    data = {"list": [mapping], "dict": mapping, "dataclass": node}[kind]
    with pytest.raises(ValueError, match=r"Cycle detected"):
        to_jsonable(data)


def test_to_jsonable_output_round_trips_through_json() -> None:
    """Test that ``json.dumps`` accepts the output."""
    data = {"a": (1, {2, 1}), 3: Node(None, [b"a"])}
    assert json.loads(json.dumps(to_jsonable(data))) == to_jsonable(data)
