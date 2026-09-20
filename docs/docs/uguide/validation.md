# Validation

:book: This page describes the `coola.validation` package, which provides a small set of
functions to validate function/method arguments and raise a standard exception with a clear
error message when a constraint is not met.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.validation` package provides:

1. **Type checks** - `validate_isinstance` to check a value is an instance of a given type
2. **Container checks** - `validate_not_empty` to check a value is not empty, and `validate_in`
   to check a value belongs to a set of valid values
3. **Comparison checks** - `validate_ge`, `validate_gt`, `validate_le`, and `validate_lt` to
   check a value against a bound

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
TypeError: count must be an instance of <class 'int'>, got <class 'str'>

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

## See Also

- [`coola.identifier`](identifier.md): Uses a similar internal validation helper module for its
  generators.
