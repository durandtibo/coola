# Architecture of `coola.formatting`

:book: This page describes the design of the `coola.formatting` package for contributors. The functions are described in the [user guide](../uguide/formatting.md) and in the [reference](../refs/formatting.md).

## Overview

`coola.formatting` turns values into text for messages, `repr` and reports. It is a **core** package
(see [Package layering](architecture.md#package-layering)): it imports `coola.utils` and
`coola.validation` only, and the packages that print values (`display`, `registry`, the error messages
of the feature packages) import it. Every function that formats a value for a message, a `repr` or a
report lives here, so there is one place to look for it.

```
coola/formatting/
    __init__.py     docstring only, no re-export
    structure.py    str_/repr_ of sequences and mappings, str_indent, repr_indent
    quantity.py     str_human_byte_size, find_best_byte_unit, str_time_human, make_bar
    text.py         char_diff_summary, count_lines, remove_empty_lines, slugify, truncate_str
    secret.py       mask_secret
    diff.py         unified_diff
```

## Rules

- **Pure and total.** A formatting function has no side effect, keeps no state and gives the same
  string for the same input. It does not touch the logger, the environment or the filesystem.
- **`str_*` and `repr_*` pairs.** The structure functions come in pairs that format the values with
  `str` or with `repr`: `str_mapping` / `repr_mapping`, `str_sequence` / `repr_sequence`, and the
  `*_line` variants that write on one line. The pair shares one private helper, so the layout is
  written once. `InlineDisplayMixin` and `MultilineDisplayMixin` of `coola.display` build their
  `repr` and `str` with them.
- **One layout.** A mapping is `(key): value` per line, a sequence is `(index): value` per line, and a
  value whose text has several lines is indented under its key (`str_indent`, `repr_indent`).
- **Strict arguments.** Options are keyword-only and checked with `coola.validation` (for example
  `validate_ge(num_spaces, 0, name="num_spaces")`).
- **No optional library.** Nothing here imports NumPy, torch or another optional package.

## Why a package of its own

The formatting functions were spread over four modules of `coola.utils` (`format`, `string`, `secret`,
`text_diff`), and `display` and `registry` imported the first. A package next to `utils` makes them
discoverable and keeps `utils` the foundation. The name is `formatting` because `coola.format` would
shadow the builtin in `from coola import format`. The sequence and mapping functions stay in one module
because they share `str_indent`.

## Adding a function

Put a function here if it turns a value into text and is useful to more than one package. A function
that is a step of one feature stays in that feature. Give it a Google docstring with a `pycon` example,
a test in `tests/unit/formatting/test_<module>.py`, and a line in `refs/formatting.md`.
