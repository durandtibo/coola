r"""Atheris fuzz harness for ``coola.summarize``.

Feeds pseudo-random nested Python objects into ``summarize`` to look for
crashes, unhandled exceptions, or infinite recursion in the summarization
logic.

Run locally with:

    pip install atheris
    python fuzz/fuzz_summarize.py
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from coola.summary import summarize

from fuzz_objects_are_equal import _build_object


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    obj = _build_object(fdp)
    max_depth = fdp.ConsumeIntInRange(0, 8)

    try:
        summarize(obj, max_depth=max_depth)
    except (TypeError, ValueError, RecursionError):
        # Expected error types for malformed/unsupported inputs.
        pass


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
