from __future__ import annotations

import subprocess
import sys

import pytest

SCRIPT = """
from dataclasses import dataclass
from coola.nested import convert_to_jsonable

@dataclass
class P:
    tags: set
    corner: tuple

print(convert_to_jsonable({"p": P({"d", "a", "b", "c", "e"}, (1, 2)), "s": {"x", "y", 1, 2}}))
"""


def run(seed: str) -> str:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT],
        env={"PYTHONHASHSEED": seed, "PATH": ""},
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.mark.parametrize("seed", ["0", "1", "2", "3", "42"])
def test_convert_to_jsonable_is_the_same_for_every_hash_seed(seed: str) -> None:
    """Test that the output does not depend on ``PYTHONHASHSEED``."""
    assert run(seed) == (
        "{'p': {'tags': ['a', 'b', 'c', 'd', 'e'], 'corner': (1, 2)}, 's': [1, 2, 'x', 'y']}\n"
    )
