r"""Atheris fuzz harness for ``coola.objects_are_equal``.

Feeds pseudo-random nested Python objects (built from raw fuzzer bytes)
into ``objects_are_equal``/``objects_are_allclose`` to look for crashes,
unhandled exceptions, or infinite recursion in the comparison logic.

Run locally with:

    pip install atheris
    python fuzz/fuzz_objects_are_equal.py
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from coola.equality import objects_are_allclose, objects_are_equal


def _build_object(fdp: atheris.FuzzedDataProvider, depth: int = 0):
    r"""Build a pseudo-random, possibly-nested Python object from fuzzer
    bytes."""
    if depth >= 5:
        return fdp.ConsumeInt(8)

    choice = fdp.ConsumeIntInRange(0, 7)
    if choice == 0:
        return None
    if choice == 1:
        return fdp.ConsumeBool()
    if choice == 2:
        return fdp.ConsumeInt(8)
    if choice == 3:
        return fdp.ConsumeFloat()
    if choice == 4:
        return fdp.ConsumeUnicode(16)
    if choice == 5:
        size = fdp.ConsumeIntInRange(0, 4)
        return [_build_object(fdp, depth + 1) for _ in range(size)]
    if choice == 6:
        size = fdp.ConsumeIntInRange(0, 4)
        return tuple(_build_object(fdp, depth + 1) for _ in range(size))
    size = fdp.ConsumeIntInRange(0, 4)
    return {fdp.ConsumeUnicode(8): _build_object(fdp, depth + 1) for _ in range(size)}


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    obj1 = _build_object(fdp)
    obj2 = _build_object(fdp)

    try:
        objects_are_equal(obj1, obj2)
        objects_are_equal(obj1, obj1)
        objects_are_allclose(obj1, obj2)
    except (TypeError, ValueError, RecursionError):
        # Expected error types for malformed/unsupported comparisons.
        pass


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
