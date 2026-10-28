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
   `validate_tuple`, `validate_dict`, `validate_mapping`, `validate_set` and `validate_frozenset`
   for the common builtin types, and `validate_number` to check a value is a real number
2. **Container checks** - `validate_not_empty` to check a value is not empty, and `validate_in`
   to check a value belongs to a set of valid values
3. **Comparison checks** - `validate_ge`, `validate_gt`, `validate_le`, and `validate_lt` to
   check a value against a bound, and `validate_open_range`, `validate_closed_range` and
   `validate_half_open_range` to check a value is in a range
4. **Sign and finiteness checks** - `validate_positive`, `validate_non_negative`,
   `validate_negative`, `validate_non_positive` and `validate_finite`

The [Properties](#properties) section lists the rules that every function follows.

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

`validate_number` checks that a value is a real number (a `numbers.Real`). It accepts `int`,
`float`, `fractions.Fraction` and the NumPy scalars, and rejects `bool`, `decimal.Decimal`,
`None` and `str`:

```pycon
>>> from coola.validation import validate_number
>>> validate_number(1)
>>> validate_number(0.5)
>>> validate_number("abc", name="rtol")
Traceback (most recent call last):
    ...
TypeError: rtol must be a real number, got <class 'str'>

```

The type checks are strict, they do not convert the value:

- `bool` is rejected by `validate_int` and `validate_number`, but it is accepted by
  `validate_isinstance(x, int)` and by the comparisons (`validate_ge(True, 0)` is valid).
- NumPy scalars are accepted by `validate_number`. `validate_int` rejects `numpy.int64`, and
  `validate_float` rejects `numpy.float32` but accepts `numpy.float64`, which subclasses `float`.
  Use `validate_isinstance(x, numpy.integer)` or `validate_number` to accept them.

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

The comparisons are duck-typed: they accept any value that supports the comparison (`Decimal`,
`date`, ...). A value that cannot be compared raises a `TypeError`. A `NaN` is never valid, since
every comparison with a `NaN` is false: it raises a `ValueError`.

`validate_open_range` checks `low < value < high`, `validate_closed_range` checks
`low <= value <= high` and `validate_half_open_range` checks `low <= value < high`:

```pycon
>>> from coola.validation import (
...     validate_closed_range,
...     validate_half_open_range,
...     validate_open_range,
... )
>>> validate_open_range(0.5, 0, 1)
>>> validate_closed_range(1, 0, 1)
>>> validate_half_open_range(0, 0, 1)
>>> validate_half_open_range(1, 0, 1, name="ratio")
Traceback (most recent call last):
    ...
ValueError: ratio must be in the half-open range [0, 1), got 1

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

| Property                   | Statement                                                                                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Silent on success          | Every function returns `None` when the value is valid. It never returns or converts the value.                                                                  |
| Standard exceptions        | A wrong type raises `TypeError`. A wrong value raises `ValueError`. No custom exception class is involved.                                                     |
| Informative message        | The message names the argument (`name`, default `"value"`), the constraint and the value received.                                                             |
| Pure                       | Validators do not modify the value and have no side effect.                                                                                                    |
| Keyword-only `name`        | `name` can only be passed by keyword, so a call cannot mix it up with a bound.                                                                                 |
| Strict types, duck compare | The type checks are strict (no conversion, `bool` is not accepted by `validate_int` and `validate_number`). The comparisons accept whatever supports `<`, `>`. |
| NaN is never valid         | A comparison with a `NaN` is false, so the bound and range checks raise `ValueError` on a `NaN`, and `validate_finite` rejects it.                             |
| Core only                  | The package imports the standard library only. NumPy scalars are supported through the `numbers` ABCs.                                                         |

## See Also

- [`coola.identifier`](identifier.md): Uses these validators to check the options of its
  generators.
