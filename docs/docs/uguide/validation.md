# Validation

:book: This page describes the `coola.validation` package, which provides a small set of
functions to validate function/method arguments and raise a standard exception with a clear
error message when a constraint is not met.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.validation` package provides:

1. **Type checks** - `validate_isinstance` to check a value is an instance of a given type, and
   `validate_bool`, `validate_int`, `validate_float`, `validate_str`, `validate_list`,
   `validate_tuple`, `validate_dict`, `validate_mapping`, `validate_set` and `validate_frozenset` for the common builtin types
2. **Container checks** - `validate_not_empty` to check a value is not empty, and `validate_in`
   to check a value belongs to a set of valid values
3. **Comparison checks** - `validate_ge`, `validate_gt`, `validate_le`, and `validate_lt` to
   check a value against a bound, and `validate_range` to check a value is in a
   range
4. **Sign and finiteness checks** - `validate_positive`, `validate_non_negative`,
   `validate_negative`, `validate_non_positive` and `validate_finite`

Every function takes the value to validate as its first argument, and an optional `name` keyword
argument used to identify the value in the error message. On success, the function returns
`None`; on failure, it raises an exception.

## Type

```pycon
>>> from coola.validation import validate_isinstance
>>> validate_isinstance(1, int)
>>> validate_isinstance(1, (int, float))
>>> validate_isinstance("abc", int, name="count")
Traceback (most recent call last):
    ...
TypeError: count must be an instance of int, got str

```

## Container

```pycon
>>> from coola.validation import validate_not_empty
>>> validate_not_empty([1, 2, 3])
>>> validate_not_empty([], name="my_list")
Traceback (most recent call last):
    ...
ValueError: my_list must not be empty

```

```pycon
>>> from coola.validation import validate_in
>>> validate_in("a", ("a", "b", "c"))
>>> validate_in("d", ("a", "b", "c"), name="mode")
Traceback (most recent call last):
    ...
ValueError: mode must be one of ('a', 'b', 'c'), got 'd'

```

## Comparison

`validate_ge`/`validate_le` check inclusive bounds, and `validate_gt`/`validate_lt` check
exclusive bounds:

```pycon
>>> from coola.validation import validate_ge, validate_gt, validate_le, validate_lt
>>> validate_ge(1, 0)
>>> validate_gt(1, 0)
>>> validate_le(0, 1)
>>> validate_lt(0, 1)
>>> validate_gt(1, 1, name="count")
Traceback (most recent call last):
    ...
ValueError: count must be greater than 1, got 1

```

`validate_range` checks `low <= value <= high`. The flags `low_inclusive` and `high_inclusive`
exclude a bound, which gives an open range or a half-open range. A range that no value can satisfy
(for example `low > high`) is a bug of the caller and raises a `ValueError` about the bounds:

```pycon
>>> from coola.validation import validate_range
>>> validate_range(1, 0, 1)
>>> validate_range(0.5, 0, 1, low_inclusive=False, high_inclusive=False)
>>> validate_range(0, 0, 1, high_inclusive=False)
>>> validate_range(1, 0, 1, high_inclusive=False, name="ratio")
Traceback (most recent call last):
    ...
ValueError: ratio must be in the range [0, 1), got 1
>>> validate_range(0.5, 1, 0, name="ratio")
Traceback (most recent call last):
    ...
ValueError: ratio: the range [1, 0] is empty

```

## Sign and Finiteness

`validate_positive`/`validate_negative` check strict signs, and `validate_non_negative`/
`validate_non_positive` accept zero. `validate_finite` rejects infinite values and `NaN`:

```pycon
>>> from coola.validation import (
...     validate_finite,
...     validate_negative,
...     validate_non_negative,
...     validate_non_positive,
...     validate_positive,
... )
>>> validate_positive(1)
>>> validate_non_negative(0)
>>> validate_negative(-1)
>>> validate_non_positive(0)
>>> validate_finite(1.5)
>>> validate_positive(0, name="count")
Traceback (most recent call last):
    ...
ValueError: count must be positive, got 0
>>> validate_finite(float("inf"), name="lr")
Traceback (most recent call last):
    ...
ValueError: lr must be finite, got inf

```

## Properties

| Property            | Statement                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Silent on success   | Every function returns `None` when the value is valid.                                                     |
| Standard exceptions | A wrong type raises `TypeError`. A wrong value raises `ValueError`. No custom exception class is involved. |
| Informative message | The message names the argument (`name`, default `"value"`), the constraint and the value received.         |
| Pure                | Validators do not modify the value and have no side effect.                                                |
| Keyword-only `name` | `name` can only be passed by keyword, so a call cannot mix it up with a bound.                             |

## See Also

- [`coola.identifier`](identifier.md): Uses a similar internal validation helper module for its
  generators.
