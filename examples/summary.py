r"""Show examples of the ``coola.summary`` and ``coola.reducer`` functions.

Run with: ``python examples/summary.py``
"""

from __future__ import annotations

import sys

from coola.reducer import NativeReducer
from coola.summary import summarize


def show(text: object) -> None:
    r"""Write a line on the standard output."""
    sys.stdout.write(f"{text}\n")


def main() -> None:
    r"""Run the examples."""
    show(summarize({"a": 1, "b": [1, 2, 3]}, max_depth=2))
    reducer = NativeReducer()
    show(f"median: {reducer.median([3, 1, 2])}")
    show(f"max: {reducer.max([3, 1, 2])}")


if __name__ == "__main__":
    main()
