# Formatting

:book: This page describes the `coola.formatting` package, which turns values into text for messages,
`repr` and reports.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

| Module                       | Content                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `coola.formatting.structure` | `str_*` and `repr_*` of sequences and mappings, indentation                         |
| `coola.formatting.quantity`  | Human-readable byte sizes and durations, `make_bar`                                 |
| `coola.formatting.text`      | `count_lines`, `remove_empty_lines`, `slugify`, `truncate_str`, `char_diff_summary` |
| `coola.formatting.secret`    | `mask_secret`                                                                       |
| `coola.formatting.diff`      | `unified_diff`                                                                      |

`coola.formatting` is a core package: it imports `coola.utils` and `coola.validation` only, and the
packages that print values (`coola.display`, `coola.registry`, the error messages of the feature
packages) are built on it.

## Properties

| Property               | Statement                                                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pure                   | A function has no side effect and keeps no state: the same input gives the same string. It does not touch the logger, the environment or files.      |
| `str` and `repr` pairs | `str_mapping` / `repr_mapping` and `str_sequence` / `repr_sequence` (and the `*_line` variants) format the values with `str` or with `repr`.         |
| One layout             | A mapping is `(key): value` per line, a sequence is `(index): value` per line, and a multi-line value is indented under its key.                     |
| Strict options         | Options are keyword-only and checked before any work (`num_spaces` must be at least 0): a wrong type is a `TypeError`, a wrong value a `ValueError`. |
| No optional library    | Nothing here imports NumPy, torch or another optional package.                                                                                       |

The exact layout of a bar or a diff is meant to be read, not parsed.

## Formatting Utilities (`coola.formatting`)

### Formatting Sequences

Use `str_sequence()` and `repr_sequence()` to format sequences:

```pycon
>>> from coola.formatting.structure import str_sequence, repr_sequence
>>> print(str_sequence([1, 2, 3, 4, 5]))
(0): 1
(1): 2
(2): 3
(3): 4
(4): 5
>>> print(repr_sequence([1, 2, 3, 4, 5]))
(0): 1
(1): 2
(2): 3
(3): 4
(4): 5

```

For single-line formatting, use `str_sequence_line()`:

```pycon
>>> from coola.formatting.structure import str_sequence_line
>>> str_sequence_line([1, 2, 3, 4, 5])
'1, 2, 3, 4, 5'

```

### Formatting Mappings

Use `str_mapping()` and `repr_mapping()` to format dictionaries:

```pycon
>>> from coola.formatting.structure import str_mapping, repr_mapping
>>> data = {"name": "Alice", "age": 30, "city": "NYC"}
>>> print(str_mapping(data))
(name): Alice
(age): 30
(city): NYC
>>> print(repr_mapping(data))
(name): Alice
(age): 30
(city): NYC

```

Sort keys before formatting:

```pycon
>>> from coola.formatting.structure import str_mapping
>>> data = {"z": 1, "a": 2, "m": 3}
>>> print(str_mapping(data, sorted_keys=True))
(a): 2
(m): 3
(z): 1

```

For single-line formatting, use `str_mapping_line()`:

```pycon
>>> from coola.formatting.structure import str_mapping_line
>>> str_mapping_line({"a": 1, "b": 2})
'a=1, b=2'

```

### Adding Indentation

Use `str_indent()` and `repr_indent()` to add indentation to multi-line strings:

```pycon
>>> from coola.formatting.structure import str_indent
>>> multi_line = "line1\nline2\nline3"
>>> print(str_indent(multi_line, num_spaces=4))
line1
    line2
    line3

```

### Human-Readable Byte Sizes

Use `str_human_byte_size()` to convert byte sizes to human-readable format:

```pycon
>>> from coola.formatting.quantity import str_human_byte_size
>>> str_human_byte_size(512)
'512.00 B'
>>> str_human_byte_size(23456)
'22.91 KB'

```

Find the best byte unit for a size:

```pycon
>>> from coola.formatting.quantity import find_best_byte_unit
>>> find_best_byte_unit(100)
'B'
>>> find_best_byte_unit(2000000)
'MB'

```

### Human-Readable Time Durations

Use `str_time_human()` to format time durations:

```pycon
>>> from coola.formatting.quantity import str_time_human
>>> import datetime
>>> str_time_human(datetime.timedelta(seconds=90).seconds)
'0:01:30'
>>> str_time_human(datetime.timedelta(hours=2, minutes=30).seconds)
'2:30:00'

```

### Progress Bars

`make_bar()` draws a bar for a fraction between 0 and 1:

```pycon
>>> from coola.formatting.quantity import make_bar
>>> make_bar(0.5)
'[█████░░░░░]'
>>> make_bar(0.25, length=8)
'[██░░░░░░]'

```

## String Utilities (`coola.formatting.text`)

The `coola.formatting.text` module provides small helpers to work with strings: `count_lines()`,
`remove_empty_lines()`, `slugify()`, `truncate_str()` and `char_diff_summary()`:

```pycon
>>> from coola.formatting.text import (
...     char_diff_summary,
...     count_lines,
...     remove_empty_lines,
...     slugify,
...     truncate_str,
... )
>>> count_lines("a\nb\nc")
3
>>> remove_empty_lines("a\n\nb")
'a\nb'
>>> slugify("Hello, World! 2024")
'hello-world-2024'
>>> truncate_str("abcdefghij", max_len=8)
'abcde...'
>>> char_diff_summary("abc", "abcdef")
'3 -> 6 chars (+3 chars, +100.0%).'

```

## Secret Utilities (`coola.formatting.secret`)

`mask_secret()` masks the content of a secret with `*` before displaying it. Only the first
`show_first` and the last `show_last` characters are kept, and a short secret is fully masked:

```pycon
>>> from coola.formatting.secret import mask_secret
>>> mask_secret("sk-abcdefghijklmnop")
'sk-************mnop'
>>> mask_secret("short")
'*****'

```

## Diff Utilities (`coola.formatting.diff`)

`unified_diff()` returns the unified diff of two texts:

```pycon
>>> from coola.formatting.diff import unified_diff
>>> print(unified_diff("a\nb\n", "a\nc\n"), end="")
--- previous
+++ latest
@@ -1,2 +1,2 @@
 a
-b
+c

```

## See Also

- [Utility Functions](utils.md)
- [Display Utilities](display.md): the mixins that build `repr` and `str` with these functions
- [Architecture of `coola.formatting`](../dev/architecture/formatting.md)
- [API reference](../refs/formatting.md)
