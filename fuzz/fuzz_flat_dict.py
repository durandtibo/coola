r"""Atheris fuzz harness for
``coola.nested.to_flat_dict``/``from_flat_dict``.

Feeds pseudo-random nested Python objects into ``to_flat_dict`` and feeds
pseudo-random flat dicts into ``from_flat_dict`` to look for crashes,
unhandled exceptions, or infinite recursion in the flatten/unflatten logic.

Run locally with:

    pip install atheris
    python fuzz/fuzz_flat_dict.py
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from coola.nested import from_flat_dict, to_flat_dict

from fuzz_objects_are_equal import _build_object


def _build_flat_dict(fdp: atheris.FuzzedDataProvider) -> dict:
    size = fdp.ConsumeIntInRange(0, 6)
    return {fdp.ConsumeUnicode(12): _build_object(fdp) for _ in range(size)}


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    separator = fdp.ConsumeUnicode(2) or "."

    nested = _build_object(fdp)
    try:
        flat = to_flat_dict(nested, separator=separator)
        from_flat_dict(flat, separator=separator)
    except (TypeError, ValueError, RecursionError):
        pass

    flat_dict = _build_flat_dict(fdp)
    try:
        from_flat_dict(flat_dict, separator=separator)
    except (TypeError, ValueError, RecursionError):
        pass


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
