# Validation

:book: This page describes the `coola.validation` package, which provides a small set of
functions to validate function/method arguments and raise a standard exception with a clear
error message when a constraint is not met.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The package has eleven functions in three families:

| Family     | Functions                                                        | Checks what a value...                    |
| ---------- | ---------------------------------------------------------------- | ----------------------------------------- |
| Type       | `validate_isinstance`, `validate_int`, `validate_number`         | **is**                                    |
| Comparison | `validate_ge`, `validate_gt`, `validate_le`, `validate_lt`, `validate_range`, `validate_finite` | **compares** to (a bound, a range, `inf`) |
| Container  | `validate_not_empty`, `validate_in`                              | **holds**, or belongs to                  |

The [Properties](#properties) section lists the rules that every function follows.

Every function takes the value to validate as its first argument, and an optional `name` keyword
argument used to identify the value in the error message. On success, the function returns
`None`; on failure, it raises an exception. The [Properties](#properties) section lists the rules
that every function follows.

A validator exists only if it states a constraint that `isinstance` or one comparison cannot
state without a special case. `validate_int` exists for the `bool` case, `validate_number` for the
`numbers.Real` plus `bool` case, and `validate_finite` for the overflow of `math.isfinite` and
the `NaN`. There is no `validate_str` (it is `validate_isinstance(x, str)`) and no
`validate_positive` (it is `validate_gt(x, 0)`).

Validators are not a schema library: one value, one constraint. To combine constraints, call
several validators; the first error is raised.

```pycon
>>> from coola.validation import validate_gt, validate_int
>>> count = 3
>>> validate_int(count, name="count")
>>> validate_gt(count, 0, name="count")

```

## Type

```pycon
>>> from coola.validation import validate_isinstance
>>> validate_isinstance(1, int)
>>> validate_isinstance(1, (int, float))
>>> validate_isinstance(1, int | float)
>>> validate_isinstance("abc", int, name="count")
Traceback (most recent call last):
    ...
TypeError: count must be an instance of int, got str
>>> validate_isinstance(1.5, int | str, name="count")
Traceback (most recent call last):
    ...
TypeError: count must be an instance of int or str, got float

```

`validate_int` rejects `bool`, and `validate_number` checks that a value is a real number (a
`numbers.Real`) and rejects `bool`:

```pycon
>>> from coola.validation import validate_int, validate_number
>>> validate_int(1)
>>> validate_number(0.5)
>>> validate_int(True, name="count")
Traceback (most recent call last):
    ...
TypeError: count must be an int, got bool
>>> validate_number("abc", name="rtol")
Traceback (most recent call last):
    ...
TypeError: rtol must be a real number, got str

```

The type checks are strict, they do not convert the value:

- `bool` is rejected by `validate_int` and `validate_number`, but it is accepted by
  `validate_isinstance(x, int)` and by the comparisons (`validate_ge(True, 0)` is valid).
- NumPy scalars are accepted by `validate_number` (they register with `numbers.Real`, so
  `coola.validation` does not import NumPy), and `Fraction` too. `decimal.Decimal` is rejected, it
  is not a `numbers.Real`.
- `validate_int` rejects `numpy.int64`. Use `validate_isinstance(x, (int, numpy.integer))`
  or `validate_number` to accept it.
- `validate_isinstance(x, float)` accepts a `float` subclass, hence a `numpy.float64` but not a
  `numpy.float32`.

## Container

```pycon
>>> from coola.validation import validate_not_empty
>>> validate_not_empty([1, 2, 3])
>>> validate_not_empty([], name="my_list")
Traceback (most recent call last):
    ...
ValueError: my_list must not be empty

```

An object with no length (an `int`, a generator) is a `TypeError`. A generator is never consumed.

```pycon
>>> from coola.validation import validate_in
>>> validate_in("a", ("a", "b", "c"))
>>> validate_in("d", ("a", "b", "c"), name="mode")
Traceback (most recent call last):
    ...
ValueError: mode must be one of ('a', 'b', 'c'), got 'd'

```

A value that cannot be tested (an unhashable value against a `set`) is not valid: it is a
`ValueError`, like any other value that is not in `valid`. A `valid` that is not a container is a
bug of the caller and its `TypeError` propagates. The listing of `valid` in the message is limited
to 20 items, so the message stays short. The order of a `set` is not stable: pass a `tuple` for a
stable message.

## Comparison

`validate_ge`/`validate_le` check inclusive bounds, and `validate_gt`/`validate_lt` check
exclusive bounds. The sign checks are comparisons with zero:

```pycon
>>> from coola.validation import validate_ge, validate_gt, validate_le, validate_lt
>>> validate_ge(1, 0)
>>> validate_gt(1, 0)
>>> validate_le(0, 1)
>>> validate_lt(0, 1)
>>> validate_gt(0, 0, name="count")
Traceback (most recent call last):
    ...
ValueError: count must be greater than 0, got 0

```

The comparisons are duck-typed: they accept any value that supports the comparison (`Decimal`,
`date`, ...). A value that cannot be compared (`None`, a `str` against a number, an array with
several items) raises a `TypeError`. A `NaN` is never valid, since every comparison with a `NaN`
is false: it raises a `ValueError`.

```pycon
>>> validate_ge(None, 0, name="count")
Traceback (most recent call last):
    ...
TypeError: count cannot be compared with 0, got None

```

`validate_range` checks `low <= value <= high`. The flags `low_inclusive` and `high_inclusive`
exclude a bound, which gives an open range or a half-open range. A range that no value can
satisfy (for example `low > high`) is a bug of the caller and raises a `ValueError` about the
bounds:

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

`validate_finite` rejects infinite values and `NaN`. It checks the type first, so a value that is
not a real number is a `TypeError`:

```pycon
>>> from coola.validation import validate_finite
>>> validate_finite(1.5)
>>> validate_finite(float("inf"), name="lr")
Traceback (most recent call last):
    ...
ValueError: lr must be finite, got inf
>>> validate_finite("a", name="lr")
Traceback (most recent call last):
    ...
TypeError: lr must be a real number, got str

```

## Properties

| Property                   | Statement                                                                                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Silent on success          | Every function returns `None` when the value is valid. It never returns or converts the value.                                                                 |
| Standard exceptions        | A wrong type raises `TypeError`. A wrong value raises `ValueError`. No custom exception class. No other exception type escapes.                                |
| Informative message        | `<name> must be <requirement>, got <value>`: the value is shown with `repr`, a type by its name. The size of the message is bounded.                           |
| Pure                       | Validators do not modify the value or the container and have no side effect. They never consume an iterator.                                                   |
| Keyword-only `name`        | `name` can only be passed by keyword, so a call cannot mix it up with a bound or a class.                                                                      |
| Strict types, duck compare | The type checks are strict (no conversion, `bool` is not accepted by `validate_int` and `validate_number`). The comparisons accept whatever supports `<`, `>`. |
| NaN is never valid         | A comparison with a `NaN` is false, so the bound and range checks raise `ValueError` on a `NaN`, and `validate_finite` rejects it.                             |
| Order of the checks        | A validator checks the type first, then the value.                                                                                                             |
| Core only                  | The package imports the standard library only.                                                                                                                 |

## Migration

The package had 26 functions. The 16 functions that were synonyms of another one are removed, with
no alias (`validate_range` is new):

| Removed                                                                                        | Replacement                               |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `validate_bool`, `validate_float`, `validate_str`, `validate_list`, `validate_tuple`, `validate_dict`, `validate_set`, `validate_frozenset` | `validate_isinstance(x, bool)`, `validate_isinstance(x, float)`, ... |
| `validate_mapping(x)`                                                                          | `validate_isinstance(x, Mapping)`         |
| `validate_positive(x)`, `validate_non_negative(x)`                                             | `validate_gt(x, 0)`, `validate_ge(x, 0)`  |
| `validate_negative(x)`, `validate_non_positive(x)`                                             | `validate_lt(x, 0)`, `validate_le(x, 0)`  |
| `validate_closed_range(x, low, high)`                                                          | `validate_range(x, low, high)`            |
| `validate_open_range(x, low, high)`                                                            | `validate_range(x, low, high, low_inclusive=False, high_inclusive=False)` |
| `validate_half_open_range(x, low, high)`                                                       | `validate_range(x, low, high, high_inclusive=False)` |

## See Also

- [`coola.identifier`](identifier.md): Uses these validators to check the options of its
  generators.
