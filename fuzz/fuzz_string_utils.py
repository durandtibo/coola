r"""Atheris fuzz harness for
``coola.utils.string``/``coola.utils.text_diff``.

Feeds pseudo-random strings into ``slugify``, ``truncate_str``,
``char_diff_summary``, and ``unified_diff`` to look for crashes or
unhandled exceptions in the string-handling utilities.

Run locally with:

    pip install atheris
    python fuzz/fuzz_string_utils.py
"""

from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from coola.utils.string import char_diff_summary, slugify, truncate_str
    from coola.utils.text_diff import unified_diff


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    text_a = fdp.ConsumeUnicode(128)
    text_b = fdp.ConsumeUnicode(128)
    max_len = fdp.ConsumeIntInRange(0, 64)
    context_lines = fdp.ConsumeIntInRange(0, 8)

    try:
        slugify(text_a)
        truncate_str(text_a, max_len=max_len)
        char_diff_summary(text_a, text_b)
        unified_diff(text_a, text_b, context_lines=context_lines)
    except (TypeError, ValueError, RecursionError):
        # Expected error types for malformed/unsupported inputs (e.g. a
        # truncate suffix longer than max_len).
        pass


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
