r"""Show examples of the ``coola.nested`` functions.

Run with: ``python examples/nested.py``
"""

from __future__ import annotations

import sys

from coola.nested import from_flat_dict, merge_mappings, to_flat_dict


def show(text: object) -> None:
    r"""Write a line on the standard output."""
    sys.stdout.write(f"{text}\n")


def main() -> None:
    r"""Run the examples."""
    data = {"model": {"layers": [{"units": 8}, {"units": 4}]}, "lr": 0.1}
    flat = to_flat_dict(data)
    show(f"flat keys: {sorted(flat)}")
    show(f"round trip: {from_flat_dict(flat, restore_lists=True) == data}")
    show(f"merged: {merge_mappings([{'a': 1}, {'b': 2}, {'a': 1}])}")


if __name__ == "__main__":
    main()
