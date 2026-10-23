r"""Show examples of the ``coola.hashing`` and ``coola.identifier`` functions.

Run with: ``python examples/hashing.py``
"""

from __future__ import annotations

import sys

from coola.hashing import hash_object
from coola.identifier import generate_stable_content_id


def show(text: object) -> None:
    r"""Write a line on the standard output."""
    sys.stdout.write(f"{text}\n")


def main() -> None:
    r"""Run the examples."""
    data = {"a": [1, 2, 3], "b": {"c": "text"}}
    same = {"b": {"c": "text"}, "a": [1, 2, 3]}
    other = {"a": [1, 2, 4], "b": {"c": "text"}}
    show(f"same hash for the same content: {hash_object(data) == hash_object(same)}")
    show(f"different hash for other content: {hash_object(data) != hash_object(other)}")
    show(f"content id has 8 characters: {len(generate_stable_content_id(data, length=8)) == 8}")


if __name__ == "__main__":
    main()
