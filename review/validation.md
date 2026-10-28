# `coola.validation`: target state and the changes to reach it

This document describes the design that `coola.validation` should have, then the changes to make to
the current implementation (`main`, `dc8dc788`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/validation.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: `validation` is a **core** package, it imports no feature package),
and it is written in the same form as `review/recursive.md`. Its aim is a package that is **small,
orthogonal and uniform**: a short list of validators, no validator that is a synonym of another one,
one error-message format, one place where a comparison is evaluated. Breaking changes are accepted,
and the future state is more important than the past one.

Evidence comes from probes run on `main` (Python 3.14, NumPy 2, scripts kept outside the repository)
and from a search of every call site in `src`. Items marked **[verified]** were reproduced;
**[reading]** comes from the code. Nothing in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 804 lines for 26 functions, 13 of which are
   one-line wrappers (nine type checks are `validate_isinstance(value, X)`, four sign checks are
   `validate_ge/gt/le/lt(value, 0)`) and three are one range with two flags. Eight of the 26
   functions have no caller in `src` (section 2). Each wrapper costs a docstring, an export, a doc entry and tests. The target
   keeps a function only if it adds behaviour that the caller would otherwise have to write.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period. The release notes list the removed names and the call-site rewrite.
3. **Core package.** `coola.validation` imports only the standard library (import-linter contract
   "Core packages must not import feature packages"). It imports NumPy or any optional library
   **never**; NumPy scalars are supported through the `numbers` ABCs they register with.
4. **Cheap.** Validators run in constructors and in the entry points of `hash_object`,
   `round_values`, `convert_to_jsonable`, ... A call costs 70 to 240 ns today **[verified]**. The
   target must not be slower on the success path (section 8, gate).
5. **One home for argument checks.** The checks that are re-implemented next to their caller
   (`validate_seed`, `validate_hash_length`, `rounding.common.validate_arguments`,
   `_validate_on_duplicate`, `_validate_sep`, `_validate_key`) use the validators of this package
   where they can (section 5), so the same mistake gives the same message everywhere.

## 1. Properties of the sub-package

These are the rules of the user guide, tightened where the code does not hold them today (section 2).
Every design choice below exists to make one of them true **by construction**: one helper builds
every message, one helper evaluates every comparison.

| Property                   | Statement                                                                                                                                                                                                                                                    | Enforced by                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| Silent on success          | Every validator returns `None` when the value is valid. It never returns the value and never converts it.                                                                                                                                                    | the signatures (`-> None`)                 |
| Standard exceptions        | A value of the wrong **type** raises `TypeError`. A value of the right type that breaks the constraint raises `ValueError`. No custom exception class. No other exception type escapes: a failing `len()`, `in` or comparison is reported as one of the two. | the helpers `_type_error` and `_compare`   |
| Informative message        | The message names the argument (`name`, default `"value"`), the constraint and the value received. The value is always shown with `repr`, a type is always shown by its name. The message size is bounded, whatever the size of the value.                   | the helpers; one test over every validator |
| Pure                       | A validator never modifies the value, the bounds or the container, and has no side effect. It never consumes an iterator it receives (it rejects it).                                                                                                        | stateless functions                        |
| Keyword-only `name`        | `name` can only be passed by keyword, so a call cannot mix it up with a bound or a class. It is the only option of every validator.                                                                                                                          | the signatures (`*, name`)                 |
| Strict types, duck compare | The **type** validators are strict: no conversion, `bool` is not an `int`, a NumPy scalar is accepted by `validate_number` only. The **value** validators are duck-typed: they accept whatever supports the comparison (`Decimal`, `date`, ...).             | the documentation of each family           |
| NaN is never valid         | A comparison with a NaN is false, so every bound and range check raises `ValueError` on a NaN, and `validate_finite` rejects it.                                                                                                                             | Python semantics, tested                   |
| Order of the checks        | A validator checks the type first, then the value. `validate_finite(x)` on a non-number is a `TypeError` that names the argument, never a raw error of `math`.                                                                                               | the implementation of each validator       |
| Core only                  | The package imports the standard library only.                                                                                                                                                                                                               | the import-linter contract                 |

Not guaranteed, and documented as such:

- **Validators are not a schema library.** One value, one constraint. Combining constraints is
  composing calls (`validate_int(x)` then `validate_gt(x, 0)`); there is no `validate_all`, no
  decorator, no error aggregation.
- **The order of the values in a message built from a `set`** (the listing of the valid values of
  `validate_in`). Pass a `tuple` for a stable message.
- **`float` subclasses are accepted** by `validate_isinstance(x, float)`, hence a `numpy.float64`
  too (it subclasses `float`) while a `numpy.float32` is not. Use `validate_number` to accept both.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Status         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **Twenty-six functions for six behaviours.** `validate_bool`, `validate_float`, `validate_str`, `validate_list`, `validate_tuple`, `validate_dict`, `validate_mapping`, `validate_set` and `validate_frozenset` are `validate_isinstance(value, X)`; `validate_positive/negative/non_negative/non_positive` are `validate_gt/lt/ge/le(value, 0)`; the three range functions differ by two booleans. Only `validate_int` (the `bool` rule) and `validate_number` add behaviour.                                                              | **[reading]**  |
| F2  | **Eight functions have no caller in `src`**: `validate_list`, `validate_tuple`, `validate_set`, `validate_frozenset`, `validate_dict`, `validate_float`, `validate_isinstance`, `validate_in`; and six more are used only by the tests: `validate_lt`, `validate_le`, `validate_negative`, `validate_non_positive`, `validate_open_range`, `validate_half_open_range`. The most used are `validate_int` (11 call sites), `validate_closed_range` (10), `validate_non_negative` (8), `validate_str` (4) and `validate_positive` (4).         | **[verified]** |
| F3  | **Three message formats for a type error.** `validate_int`: `value must be an int, got <class 'str'>`. `validate_isinstance`: `value must be an instance of <class 'int'>, got <class 'str'>` (and for a tuple `(<class 'int'>, <class 'str'>)`). `validate_number`: `value must be a real number, got <class 'str'>`. The class repr is noise; a union `int \| str` prints as is.                                                                                                                                                          | **[verified]** |
| F4  | **Raw exceptions escape, with no name.** `validate_finite("a")` and `validate_finite(None)` raise `TypeError: must be real number, not str`; `validate_finite(np.array([1., 2.]))` raises `only 0-dimensional arrays can be converted to Python scalars`; `validate_not_empty(5)` raises `object of type 'int' has no len()`; `validate_in([], {1})` raises `cannot use 'list' as a set element`; `validate_in(1, 5)` raises `argument of type 'int' is not a container or iterable`. The property "informative message" is false for them. | **[verified]** |
| F5  | **`validate_finite` accepts `True`** while `validate_number(True)` rejects it, and it accepts a `Decimal` that `validate_number` rejects. The two functions of `numeric.py` disagree on what a number is; the callers (`Tolerance`) have to call `validate_number` first to work around it.                                                                                                                                                                                                                                                 | **[verified]** |
| F6  | **The comparison error blames the type.** `validate_ge("b", "a")` passes, `validate_ge(None, 0)` raises `value must be a number that can be compared with 0, got None`. The comparisons are duck-typed (dates, strings, `Decimal` work) but the message says "number". The wording is wrong for the legitimate non-numeric uses.                                                                                                                                                                                                            | **[verified]** |
| F7  | **Two copies of the comparison logic.** `_check` and `_check_range` both contain the `try / except (TypeError, ValueError) / bool(...)` block, the `ValueError` for an array and the message construction.                                                                                                                                                                                                                                                                                                                                  | **[reading]**  |
| F8  | **An empty range is accepted silently.** `validate_closed_range(0.5, 1, 0)` raises `value must be in the closed range [1, 0], got 0.5`: the message blames the value when the bounds are wrong (a bug of the caller). Nothing can satisfy the constraint, so the error is a `ValueError` about the bounds.                                                                                                                                                                                                                                  | **[verified]** |
| F9  | **`validate_in` prints the whole container.** `validate_in(-1, list(range(100000)))` raises a `ValueError` whose message is 688,919 characters long. A `set` of valid values is printed in hash order (`{'b', 'a'}`), so the message changes from one process to the next. An iterator passed as `valid` is consumed by the test.                                                                                                                                                                                                           | **[verified]** |
| F10 | **Inconsistent `bool` and NumPy rules.** `validate_int` rejects `bool` and `numpy.int64`; `validate_float` rejects `numpy.float32` but accepts `numpy.float64` (a `float` subclass); `validate_number` accepts all NumPy scalars and rejects `bool`; the comparisons accept `True` (`validate_ge(True, 0)` passes). The rules are not written in one place.                                                                                                                                                                                 | **[verified]** |
| F11 | **Message values mix `str` and `repr`.** `validate_ge(...)` prints `got {value}` and `validate_in` prints `got {value!r}`; for a string bound the two are ambiguous (`got 1` for `"1"`). A NumPy 2 scalar prints as `np.float32(-1.0)` with `repr`, `-1.0` with `str`.                                                                                                                                                                                                                                                                      | **[verified]** |
| F12 | **The checks are re-implemented next to their callers**, with other messages: `random.seed.validate_seed` (int + range), `hashing.engine.validate_hash_length`, `rounding.common.validate_arguments` (mode in a tuple, `decimals` int and non-negative, with its own `bool` test), `nested.mapping._validate_on_duplicate` (a `validate_in`), `identifier.checksummed._validate_sep`, `registry.*._validate_key` (a `validate_isinstance`).                                                                                                 | **[reading]**  |
| F13 | **The pair "type, then bound" is written at every call site** (`validate_int(x)` then `validate_positive(x)` in 8 places: `equality.config`, `summary.config`, `identifier.checksummed`, `identifier.snowflake`, ...). It is correct and explicit; the cost is one line per option. Not a defect, recorded so that the design does not add options to merge them (decision D3).                                                                                                                                                             | **[reading]**  |
| F14 | **Four modules for twenty-six functions.** `numeric.py` (2 functions) splits what a number is (`validate_number`) from what a finite number is (`validate_finite`), and neither belongs with the comparisons or the types by the module name. `validate_finite` is a value constraint, `validate_number` a type constraint.                                                                                                                                                                                                                 | **[reading]**  |
| F15 | **The guide is incomplete.** It lists `validate_number` nowhere (the overview of section "Overview" omits it), the type section shows `validate_isinstance` only, the Properties table omits NaN handling, the `bool` rule and the NumPy rule. The "See also" points to `coola.identifier` as "a similar internal validation helper", which is no longer true (identifier uses this package).                                                                                                                                               | **[verified]** |
| F16 | **No benchmark and no property test for the value validators.** `tests/properties/test_validation.py` covers the type validators only (116 lines); the comparisons, ranges and NaN have unit tests only; there is no guard on the cost.                                                                                                                                                                                                                                                                                                     | **[reading]**  |

What is already right and must not regress: keyword-only `name`; `TypeError` for a type and
`ValueError` for a value; the `bool` rejection of `validate_int` and `validate_number`; the
`TypeError` for an incomparable value and an array with several items (not a raw `ValueError`); the
NaN behaviour of the comparisons; the `int` fast path of `validate_finite` (a huge `int` overflows
`math.isfinite`); success is silent; the `from` chain of the converted exceptions.

## 3. Architecture

Two helpers own the invariants; a validator only states its constraint.

```
validate_xxx(value, ..., *, name="value")
        |
        |-- type validators      -> _type_error(name, expected, value)       builds the TypeError
        |-- comparison validators-> _compare(test, value, bounds, name, req) evaluates, converts, builds
        |-- container validators -> _type_error / ValueError with a bounded listing
        v
None, or TypeError / ValueError with the message
        "<name> must be <requirement>, got <value!r>"
```

The message format is **one sentence**, the same for every validator:

| Case                       | Message                                                                                |
| -------------------------- | -------------------------------------------------------------------------------------- |
| wrong type                 | `<name> must be <expected>, got <type name>` (`count must be an int, got str`)         |
| wrong value                | `<name> must be <requirement>, got <value!r>` (`count must be greater than 0, got -1`) |
| not comparable             | `<name> cannot be compared with <bound>, got <value!r>`                                |
| bounds make an empty range | `<name>: the range [<low>, <high>] is empty` (a `ValueError`, section 4.3)             |

The expected type is written once, by `_type_name`: a class gives its `__qualname__` (`int`), a tuple
or a union gives `int or str`, an ABC gives its name (`Mapping`). `validate_int` and
`validate_number` pass a phrase (`an int`, `a real number`) because the article matters.

### 3.1 The helpers

```python
def _type_error(value, expected: str, name: str) -> NoReturn:
    msg = f"{name} must be {expected}, got {type(value).__qualname__}"
    raise TypeError(msg)


def _compare(test: Callable[[Any], Any], value, bounds: str, name: str, requirement: str) -> None:
    try:
        is_valid = bool(test(value))
    except (TypeError, ValueError) as exc:  # incomparable, or the truth of an array is ambiguous
        msg = f"{name} cannot be compared with {bounds}, got {value!r}"
        raise TypeError(msg) from exc
    if not is_valid:
        msg = f"{name} must be {requirement}, got {value!r}"
        raise ValueError(msg)
```

`_compare` replaces `_check` **and** `_check_range` (F7). The four bound checks, the range and the
finiteness check call it with a `lambda`/`operator.partial` test; the closure costs about 40 ns
(section 6), below the margin of the gate. Both helpers stay private in `coola.validation`: they are
an implementation detail, and a package that wants the same format writes a call to a validator.

### 3.2 The functions (target API)

`from coola.validation import ...` (14 names, down from 26):

| Family    | Function                                                                             | Raises                                   | Notes                                                                                                                |
| --------- | ------------------------------------------------------------------------------------ | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Type      | `validate_isinstance(value, cls, *, name)`                                           | `TypeError`                              | general; `cls` is a type, a tuple of types or a union; `bool` is an `int` here (documented)                          |
| Type      | `validate_int(value, *, name)`                                                       | `TypeError`                              | the only shortcut with its own rule: `bool` rejected; `numpy.int64` rejected (strict, decision D2)                   |
| Type      | `validate_number(value, *, name)`                                                    | `TypeError`                              | `numbers.Real`, `bool` rejected; NumPy scalars, `Fraction` accepted; `Decimal` rejected (not a `Real`)               |
| Bound     | `validate_ge`, `validate_gt`, `validate_le`, `validate_lt` `(value, bound, *, name)` | `TypeError` (incomparable), `ValueError` | duck-typed; the sign checks are `validate_ge(x, 0)` ...                                                              |
| Range     | `validate_range(value, low, high, *, low_inclusive=True, high_inclusive=True, name)` | `TypeError`, `ValueError`                | replaces `open`, `closed` and `half_open`; closed by default (the form all 10 call sites of the range functions use) |
| Finite    | `validate_finite(value, *, name)`                                                    | `TypeError` (not a number), `ValueError` | calls `validate_number` first (F5): `bool`, `None`, `str`, arrays give a named `TypeError`                           |
| Container | `validate_not_empty(value, *, name)`                                                 | `TypeError` (no `len`), `ValueError`     | `len(value) == 0`; an iterator or a generator has no `len` and is a `TypeError`, not consumed                        |
| Container | `validate_in(value, valid, *, name)`                                                 | `ValueError`                             | an unhashable value is not in a `set`: `ValueError`, not a raw `TypeError`; listing bounded (section 4.4)            |

Removed: `validate_bool`, `validate_float`, `validate_str`, `validate_list`, `validate_tuple`,
`validate_dict`, `validate_mapping`, `validate_set`, `validate_frozenset` (replaced by
`validate_isinstance(value, X)`, F1); `validate_positive`, `validate_non_negative`,
`validate_negative`, `validate_non_positive` (replaced by `validate_gt/ge/lt/le(value, 0)`);
`validate_open_range`, `validate_closed_range`, `validate_half_open_range` (replaced by
`validate_range`). Kept as they are: `validate_isinstance`, `validate_int`, `validate_number`,
`validate_ge`, `validate_gt`, `validate_le`, `validate_lt`, `validate_finite`, `validate_not_empty`,
`validate_in`.

The rule to add a function in the future (written in the module docstring and in the guide): _a
validator exists only if it states a constraint that `isinstance` or one comparison cannot state
without a special case_. `validate_int` exists for the `bool` case; `validate_number` for the
`numbers.Real` plus `bool` case; `validate_finite` for the overflow of `math.isfinite` and the
NaN. `validate_str` does not: `validate_isinstance(x, str)` is the same line.

### 3.3 Modules

Three modules instead of four, named by what the function checks:

| Module          | Contains                                                                | Why                                                      |
| --------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- |
| `type.py`       | `_type_error`, `validate_isinstance`, `validate_int`, `validate_number` | what a value **is**                                      |
| `comparison.py` | `_compare`, `validate_ge/gt/le/lt`, `validate_range`, `validate_finite` | what a value **compares** to (finite = `-inf < x < inf`) |
| `container.py`  | `validate_not_empty`, `validate_in`                                     | what a value **holds** or belongs to                     |

`numeric.py` is deleted (F14). `__init__.py` re-exports the 14 names; the order of `__all__` stays
alphabetical. `docs/docs/refs/validation.md` already lists three submodules and is unchanged.

## 4. Behaviour in detail

### 4.1 Types

```python
validate_isinstance(value, cls, *, name="value")
#   TypeError("count must be an instance of int, got str")
#   TypeError("count must be an instance of int or str, got float")      # tuple or union
validate_int(value, *, name="value")
#   TypeError("count must be an int, got bool")
validate_number(value, *, name="value")
#   TypeError("rtol must be a real number, got NoneType")
```

- `bool` is rejected by `validate_int` and `validate_number`, accepted by
  `validate_isinstance(x, int)` and by the comparisons. The guide has one sentence with the three
  facts (F10).
- NumPy scalars: accepted by `validate_number` (registered `numbers.Real`), rejected by
  `validate_int` (strict, decision D2). `validate_isinstance(x, np.integer)` accepts them for a
  caller who wants that, with no import of NumPy in this package.
- `cls` that is not a type raises the `TypeError` of `isinstance` unchanged: it is a bug of the
  caller, not of the value.

### 4.2 Bounds and ranges

```python
validate_ge(value, low, *, name="value")                       # value >= low
validate_gt(value, low, *, name="value")                       # value >  low
validate_le(value, high, *, name="value")                      # value <= high
validate_lt(value, high, *, name="value")                      # value <  high
validate_range(value, low, high, *, low_inclusive=True, high_inclusive=True, name="value")
```

- Messages: `count must be greater than 0, got -1`; `ratio must be in the range [0, 1), got 1`
  (the bracket notation is the existing "half-open" notation, derived from the two flags; there are
  no three words to remember, `validate_range` documents the flags).
- `validate_range` checks `low <= high` (and `low < high` when a side is open and equal) **before**
  the value, and raises `ValueError("name: the range [1, 0] is empty")` (F8). The check costs one
  comparison and turns a silent caller bug into an immediate one. It is skipped when the bounds are
  not comparable with each other, which is reported by the comparison with the value as today.
- The comparisons are duck-typed: `validate_ge(date(2024, 1, 1), date(2020, 1, 1))` is valid. The
  message of an incomparable value says "cannot be compared with" (F6).
- A NaN is never in a bound or a range (the comparison is false): `ValueError`, message with `nan`.
- An array with several items is a `TypeError` (the ambiguous truth value is converted), as today.

### 4.3 Finite and number

`validate_finite(value)` is `validate_number(value)` then `math.isfinite` for non-`int` values (the
`int` fast path is kept: a huge `int` overflows `math.isfinite`). A `Decimal` is therefore a
`TypeError` (not a `numbers.Real`); the callers of `Tolerance` can drop their `validate_number`
before `validate_finite`, but keep it for clarity: the call is cheap and explicit (the order "type,
bound, finite" stays visible at the call site).

### 4.4 Containers

- `validate_not_empty(value)`: `len(value) == 0` raises
  `ValueError("name must not be empty")`. An object without `len` raises
  `TypeError("name must have a length, got int")`. A one-shot iterator is a `TypeError`: the
  validator would consume it to know, and a consumed iterator is worse than a rejected one.
- `validate_in(value, valid)`: `value not in valid` raises
  `ValueError("mode must be one of ('a', 'b', 'c'), got 'd'")`. A value that cannot be tested
  (unhashable against a `set`) is not valid: the `TypeError` of `in` is caught and reported as the
  same `ValueError`, so the caller catches one exception type. `valid` that is not a container is a
  bug of the caller and its `TypeError` propagates. The listing of `valid` uses `reprlib` with a
  limit of 20 items (`one of ('a', 'b', ... )`), so the message of F9 is a few hundred characters, and
  the listing is built only on the failure path.

## 5. Dependents

The migration of the call sites is mechanical and each rewrite is a pure rename. The messages change
(the unified format), so the tests of the callers that match a message change with them.

| Package                                                                      | Change                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `utils` (`bloom_filter`, `batching`, `text_diff`, `format`, `secret`, `lru`) | `validate_positive(x)` becomes `validate_gt(x, 0)`, `validate_non_negative(x)` becomes `validate_ge(x, 0)`. Nothing else.                                                                                                                                                                                   |
| `equality.config`, `equality.tolerance`, `summary.config`                    | Same rename. `Tolerance` keeps `validate_number`, bound, `validate_finite`.                                                                                                                                                                                                                                 |
| `identifier` (`objectid`, `ulid`, `uuid7`, `snowflake`, `checksummed`)       | `validate_closed_range(...)` becomes `validate_range(...)`; `validate_str(x)` becomes `validate_isinstance(x, str)`; `validate_positive(x)` becomes `validate_gt(x, 0)`. `_validate_sep` keeps its own checks (the Crockford rule) and calls `validate_isinstance(sep, str)`.                               |
| `nested.keys`, `nested.mapping`                                              | `validate_mapping(x)` becomes `validate_isinstance(x, Mapping)`. `_validate_on_duplicate` is deleted: `validate_in(on_duplicate, ("raise", "first", "last", "prefix"), name="on_duplicate")` (F12; the valid values become a tuple, so the message is stable). `nested.polars`, `nested.flat`: rename only. |
| `random.seed.validate_seed`                                                  | Becomes `validate_int(seed, name="seed")` then `validate_range(seed, 0, MAX_SEED, high_inclusive=False, name="seed")`, behind the same public function (it keeps its place in `coola.random`, which owns the 2**63 bound).                                                                                  |
| `hashing.engine.validate_hash_length`                                        | `validate_int` + `validate_range(length, 2, 128)`; the parity rule (`length % 2`) stays local. The hot-path fast test stays first, as today (it is measured).                                                                                                                                               |
| `rounding.common.validate_arguments`                                         | `validate_in(mode, MODES, name="mode")`, `validate_int(decimals)`, `validate_ge(decimals, 0)`: the private `bool` test and the hand-made messages are deleted (F12).                                                                                                                                        |
| `registry.registry`, `registry.type` (`_validate_key`)                       | Not changed: `registry` is a core package and a `validate_isinstance(key, type)` call would need no more code, but the message `Expected a type as key` is part of its own contract and tests. Candidate for a later item.                                                                                  |

No dependent imports anything private from `coola.validation`.

## 6. Evidence

Probes on `main` **[verified]**: the cost of a call on the success path (mean of 200,000 calls, one
run, Apple M1):

| Call                           | Time   | Remark                                                         |
| ------------------------------ | ------ | -------------------------------------------------------------- |
| `validate_int(1)`              | 67 ns  | inline `isinstance(x, int) and not isinstance(x, bool)`: 50 ns |
| `validate_float(1.0)`          | 66 ns  |                                                                |
| `validate_number(1)`           | 155 ns | `numbers.Real` is an ABC check                                 |
| `validate_finite(1.0)`         | 69 ns  |                                                                |
| `validate_positive(1)`         | 81 ns  |                                                                |
| `validate_ge(1, 0)`            | 132 ns | `_check` with a `try` and a `bool(...)`                        |
| `validate_closed_range(1,0,2)` | 242 ns | two comparisons through `operator` functions                   |

The target `validate_range` evaluates `low <= value <= high` in one expression inside the `test`
callable (a chained comparison is one bytecode sequence, and cheaper than two `operator` calls), so it
is expected to be at most the current cost; `validate_ge/gt/le/lt` keep the current cost; the extra
`low <= high` check of F8 adds about 30 ns. The gate (section 8) is "no slower than `main` beyond 10 %
on any of the seven rows above", measured in the same run. If a row regresses, the wrapper is
inlined, not the design changed.

Message probes **[verified]**: the 688,919-character message of F9; the nondeterministic set order
(`{'b', 'a'}` and `{'a', 'b'}` in different processes); the raw exceptions of F4; `Decimal('NaN')`
accepted by `validate_finite`, rejected by `validate_number`.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step                              | Change                                                                                                                                                                                                                                                                                                                                                                                    | Breaking                  |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| 1                                 | **Fix the guide (F15).** Add `validate_number`, the `bool` and NumPy rules, the NaN rule, the **Properties** table of section 1 at the top of the page and in the package docstring; fix "See also". Documentation only.                                                                                                                                                                  | no                        |
| 2                                 | **One helper, one message format (F3, F4, F6, F7, F11).** Add `_type_error` and `_compare`, rewrite `_check`/`_check_range` on `_compare`, switch the messages to `repr` and type names, wrap the raw exceptions of `validate_finite` (calls `validate_number` first, F5), `validate_not_empty` and `validate_in`, bound the listing (F9). Update the tests that match a message.         | yes (messages, `Decimal`) |
| 3                                 | **Range (F8).** Add `validate_range`, delete the three range functions, port the 10 call sites and the tests. The empty-range check.                                                                                                                                                                                                                                                      | yes                       |
| 4                                 | **Delete the synonyms (F1, F2).** Remove the nine type shortcuts and the four sign checks, port every call site (section 5) and the tests, delete `numeric.py` (F14: `validate_number` moves to `type.py`, `validate_finite` to `comparison.py`). One commit per family so that each diff is mechanical.                                                                                  | yes                       |
| 5                                 | **Use the package where the checks are re-implemented (F12).** `validate_seed`, `validate_hash_length`, `rounding.common.validate_arguments`, `_validate_on_duplicate`, `_validate_sep`. A behaviour change only in messages.                                                                                                                                                             | messages                  |
| 6                                 | **Tests and benchmark (F16).** Property tests for the value validators, the message-format test over every validator, the cost gate (section 8). Can be merged into the step that adds the code.                                                                                                                                                                                          | no                        |
| 7                                 | **Documentation.** Rewrite `uguide/validation.md` (overview by family, the one-line rule for adding a validator, the migration table), the `validation` lines of `dev/architecture.md`, a migration section in `migration/v0.11_to_v1.0.md` (removed names with the replacement line), the status table of `review/suggestions.md`, and the rule of D6 in the module docstring, the guide |
| and `CLAUDE.md` ("Code" section). | no                                                                                                                                                                                                                                                                                                                                                                                        |

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/validation/`); the conventions of `CLAUDE.md` apply
(one behaviour per test, literal expected values, a regression test for every bug, no local imports).

- **Type** (`test_type.py`): `validate_isinstance` with a class, a tuple and a union (the message
  asserted whole: `value must be an instance of int or str, got float`); `validate_int` and
  `validate_number`: `bool`, `None`, `str`, `numpy.int64`, `numpy.float32`, `Fraction`, `Decimal`; the
  `name` in every message; a subclass of `int` (accepted).
- **Comparison** (`test_comparison.py`): for each of `ge/gt/le/lt`: below, equal, above, NaN, `None`,
  a `str`, an array with several items, a `date` (duck typing), `bool`; `validate_range`: the four
  flag combinations at both bounds, inside, a NaN, an empty range, an incomparable value, bounds that
  are not comparable with each other; `validate_finite`: finite, `inf`, `-inf`, NaN, huge `int`,
  `numpy.float32('inf')`, `None`, `True`, `Decimal`, an array.
- **Container** (`test_container.py`): empty and non-empty list, `str`, `dict`, a NumPy empty array;
  an object without `len` (the named `TypeError`), a generator (not consumed: assert
  `next(gen)` still works afterwards); `validate_in`: present, absent, unhashable value against a
  `set` (a `ValueError`, regression for F4), a long container (message length under 500, regression
  for F9), a `set` of valid values (message sorted by the listing rule, or documented unstable), a
  non-container `valid` (the `TypeError` propagates).
- **Message format** (one parametrised test over every validator and one wrong input each): the
  message starts with `"<name> must be"` or `"<name> cannot be"` and contains the received value or
  type name. This is the executable form of the property "informative message".
- **Purity**: the inputs are unchanged after success and failure (a list, a `dict`, a NumPy array), an
  iterator passed to `validate_not_empty` is not consumed, the calls are deterministic.
- **Properties** (`tests/properties/test_validation.py`, extend): `validate_ge(x, y)` raises
  `ValueError` exactly when `x < y` for finite floats; `validate_range` equals the conjunction of two
  bound checks; a `float` outside `[low, high]` is a `ValueError`, inside is silent; the two flags
  match the four mathematical intervals; `validate_finite` passes every finite float and rejects
  `inf`, `-inf`, `nan`; `validate_int` rejects `bool`; the existing type tests stay for
  `validate_isinstance` and `validate_int`.
- **Integration** (`tests/integration/`): the public arguments that call a validator (one per
  dependent of section 5) raise the unified message with the real validator, no patching
  (for example `Tolerance(rtol=-1)`, `hash_object(x, length=3)`, `round_values(x, decimals=True)`,
  `merge_mappings(..., on_duplicate="x")`).
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Cost gate** (`benchmarks/`, new `test_validation_benchmark.py`): the seven rows of section 6 on
  the success path; no row slower than `main` by more than 10 %. A failing gate rejects the step.
- Coverage: 100 % of `src/coola/validation` (`--cov=coola`), no `# pragma: no cover`.

## 9. Decisions

All seven decisions are settled: D1 to D6 follow the recommendation, D7 is "no". The design above
is the decided design; the text of each decision is kept as the reason.

**D1. Delete the synonyms?** _Recommended: yes._ Thirteen of the 26 functions are a one-line wrapper
and eight have no caller (F1, F2). They are the part of the package that grows (the range functions
and the type shortcuts were added one by one) and each one needs a docstring, a doc entry, an
export and tests. The cost for a user is writing `validate_isinstance(x, str)` instead of
`validate_str(x)`, and `validate_gt(x, 0)` instead of `validate_positive(x)`; the gain is a package
where the list of names can be read in ten seconds and the rule to add one is written down.
Alternative: keep `validate_positive` and `validate_non_negative` (the two most used of the sign
checks, 12 call sites) and drop the symmetric pair; cheaper to migrate, but then `negative` is
missing and the list needs an explanation. The full set is not kept: the package is not a public toolbox for
other libraries, which the user guide does not claim.

Decision: **adopted**.

**D2. Are `numpy.int64` and `numpy.float32` ints and floats?** _Recommended: no, for the type
validators; yes through `validate_number`._ A type annotation `int` rejects `numpy.int64` for a type
checker, `validate_int` follows the annotation, and a caller that accepts NumPy scalars writes
`validate_isinstance(x, (int, np.integer))` or uses `validate_number`. The alternative is
`numbers.Integral` for `validate_int` (accepts NumPy ints, still rejects `bool` by the explicit
test); it is a one-line change, but it makes the strictness depend on the registration of the ABCs,
and an `int` argument that is later used as `range(n)` or a shift accepts `numpy.int64` anyway.
Settled: the strict rule stays; revisit only if a user reports a real case.

Decision: **adopted**.

**D3. Merge "type then bound" into one call?** _Recommended: no._ `validate_int(x, min=0)` would
remove 8 lines at the call sites (F13) and add options to the most used function, with a second
message path. Two calls are explicit, and the order of the errors (a `TypeError` for `"a"` before a
`ValueError` for `-1`) is visible. Revisit if a call site needs three constraints on one argument.

Decision: **adopted**.

**D4. Raise on an empty range (F8)?** _Recommended: yes._ `validate_range(x, 1, 0)` cannot succeed;
it is a programming error and the current message blames the value. A `ValueError` about the bounds
costs one comparison. Alternative (rejected): leave it and document, because the failing message sends
the reader to the wrong argument.

Decision: **adopted**.

**D5. `validate_in` with an unhashable value: `ValueError` or `TypeError`?** _Recommended:
`ValueError`._ From the point of view of the caller, `[]` is not one of the valid modes; the fact that
the container is a `set` is an implementation detail. A `TypeError` would be right only if `valid`
were not a container, and that case still raises (the bug of the caller). The cost is that a
`valid` that is a string with a non-string value (`1 in "abc"` raises `TypeError`) is reported as a
value error: acceptable, the message shows both.

Decision: **adopted**.

**D6. Where does the project-wide rule live?** _Recommended: in this package's docstring and the
guide, plus a sentence in `CLAUDE.md`._ "Validate an argument with `coola.validation`; do not write
`raise ValueError` for a type, range or membership check, and do not add a validator that is one
`isinstance` or one comparison." F12 shows what happens without the rule: six private copies with
their own messages.

Decision: **adopted**.

**D7. Make `validate_*` available as a decorator or an `Annotated` metadata?** _Recommended: no._ A
schema layer (pydantic, attrs validators, beartype) does it better, and none of the call sites needs
it. The package stays a set of functions that a constructor calls.

Decision: **no**, the package stays a set of functions.

## 10. Not covered

- **Aggregated errors** (all the invalid fields of a config in one exception). The call sites raise
  on the first error, as the standard library does.
- **Custom exception classes** and error codes. The two builtin types are the contract.
- **Validation of nested structures and of array shapes or dtypes.** `validate_isinstance(x,
np.ndarray)` works; a dtype check belongs to the package that owns the array library (lazy import).
- **Coercion** (`ensure_int`, `to_float`). Validators never convert (property "silent on success").
- **`registry._validate_key`**: kept local (section 5), a candidate for a later item.
- **A `validate_callable`, `validate_path`, `validate_enum`, ...**: each is
  `validate_isinstance(x, Callable)` or `validate_in(x, tuple(Enum))`; they are added only if they
  carry a rule of their own (rule of section 3.2).

## Appendix: effect on the size of the package

| Module          | Lines today | Expected after | Why                                                                                            |
| --------------- | ----------- | -------------- | ---------------------------------------------------------------------------------------------- |
| `type.py`       | 266         | about 110      | 11 functions become 3, plus `validate_number` from `numeric.py`                                |
| `comparison.py` | 362         | about 150      | 11 functions become 6 plus `validate_finite`; one `_compare` instead of two helpers            |
| `container.py`  | 55          | about 75       | the named `TypeError`, the bounded listing                                                     |
| `numeric.py`    | 60          | 0              | removed                                                                                        |
| `__init__.py`   | 61          | about 30       | 14 names                                                                                       |
| **Package**     | **804**     | **about 365**  | docstrings dominate: about 70 % of the lines are `Args`/`Raises`/`Example`                     |
| tests (unit)    | 596         | about 450      | fewer wrappers to test, more cases per validator                                               |
| call sites      | n/a         | about plus 5   | `_validate_on_duplicate`, the `bool` test of `rounding` and the hand-made messages are deleted |

The line counts are estimates from the current docstrings and from the code to delete; the gate is the
behaviour (sections 1 and 8), not the size.

## Status

| Step | Change                                   | PR    |
| ---- | ---------------------------------------- | ----- |
| 1    | Fix the guide (F15)                      | #2555 |
| 2    | One helper, one message format           | #2556 |
| 3    | `validate_range`, the empty range        | #2557 |
| 4    | Delete the synonyms, delete `numeric.py` | #2558 |
| 5    | Use the package where checks were copied | #2559 |
| 6    | Tests and benchmark, fast success path   | #2560 |
| 7    | Documentation                            | #2561 |

Differences from the design that the implementation found: the package has 11 functions, not 14
(the design counted 14 in its text, its table lists 11); the migration section is in
`migration/v1.1_to_v2.0.md` (`2.0.0a0` is the current version); `registry._validate_key` is kept
local as decided; the cost gate (section 8) needed an inline fast path in the bound checks,
`validate_range`, `validate_finite` and `validate_number`, because a closure and `_compare` on every
call were 2 to 4 times slower than `main`.
