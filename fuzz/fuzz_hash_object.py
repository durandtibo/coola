r"""Atheris fuzz harness for ``coola.hashing.hash_object``.

Feeds pseudo-random nested Python objects into ``hash_object`` to look for
crashes, unhandled exceptions, or infinite recursion in the hashing logic.

Run locally with:

    pip install atheris
    python fuzz/fuzz_hash_object.py
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from coola.hashing import hash_object

from fuzz_objects_are_equal import _build_object


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    obj = _build_object(fdp)
    length = fdp.ConsumeIntInRange(2, 128) & ~1  # must be even
    length = max(length, 2)
    ignore_unhashable = fdp.ConsumeBool()

    try:
        hash_object(obj, length=length, ignore_unhashable=ignore_unhashable)
    except (TypeError, ValueError, RecursionError):
        # Expected error types for malformed/unhashable inputs.
        pass


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
