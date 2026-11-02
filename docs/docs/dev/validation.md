# Architecture of `coola.validation`

:book: This page describes the design of the `coola.validation` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/validation.md#properties); the API is in the [reference](../refs/validation.md).

## Overview

`coola.validation` is a core package: it imports the standard library only (NumPy scalars are
supported through the `numbers` ABCs they register with). It has eleven functions in three modules,
named by what the function checks: `type.py` (`validate_isinstance`, `validate_int`,
`validate_number`), `comparison.py` (`validate_ge/gt/le/lt`, `validate_range`, `validate_finite`)
and `container.py` (`validate_not_empty`, `validate_in`). Two private helpers own the invariants:
`_type_error` builds every `TypeError` and `_compare` evaluates every comparison and builds the
`TypeError` (incomparable value) or `ValueError` (comparison false), so all the messages have one
form, `<name> must be <requirement>, got <value!r>`. The success path is inline (a plain comparison
or an exact type test) and the helpers run on the failure path only, which keeps a call as cheap as
the code it replaces (`tests/benchmarks/test_validation_benchmark.py`).

The rule for the whole project: validate an argument with `coola.validation`; do not write
`raise ValueError` for a type, range or membership check, and do not add a validator that is one
`isinstance` or one comparison (`validate_str(x)` is `validate_isinstance(x, str)`,
`validate_positive(x)` is `validate_gt(x, 0)`). See the
[user guide](../uguide/validation.md#properties) for the properties of the package.

## Design constraints

1. **Simple and easy to maintain.** A function is kept only if it adds behaviour that the caller would
   otherwise have to write. A wrapper such as `validate_str(x)` (which is `validate_isinstance(x, str)`)
   or `validate_positive(x)` (which is `validate_gt(x, 0)`) would cost a docstring, an export, a doc
   entry and tests each, and the list of names would grow one wrapper at a time.
2. **Core package.** `coola.validation` imports only the standard library (import-linter contract
   "Core packages must not import feature packages"). It imports NumPy or any optional library
   **never**; NumPy scalars are supported through the `numbers` ABCs they register with.
3. **Cheap.** Validators run in constructors and in the entry points of `hash_object`,
   `round_values`, `to_jsonable`, ... A call costs 70 to 240 ns, and the success path must not be
   slower.
4. **One home for argument checks.** The checks that sit next to their caller (`validate_seed`,
   `validate_hash_length`, `rounding.common.validate_arguments`, `_validate_on_duplicate`,
   `_validate_sep`) use the validators of this package where they can, so the same mistake gives the
   same message everywhere.

## Architecture in detail

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
| bounds make an empty range | `<name>: the range [<low>, <high>] is empty` (a `ValueError`)                          |

The expected type is written once, by `_type_name`: a class gives its `__qualname__` (`int`), a tuple
or a union gives `int or str`, an ABC gives its name (`Mapping`). `validate_int` and
`validate_number` pass a phrase (`an int`, `a real number`) because the article matters.

### The helpers

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

`_compare` is the one evaluation. The four bound checks, the range and the finiteness check call it
with a `lambda`/`operator.partial` test; the closure costs about 40 ns, below the margin of the gate.
Both helpers are private to `coola.validation`: they are an implementation detail, and a package that
wants the same format writes a call to a validator.

### The functions

`from coola.validation import ...` (14 names):

| Family    | Function                                                                             | Raises                                   | Notes                                                                                                  |
| --------- | ------------------------------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Type      | `validate_isinstance(value, cls, *, name)`                                           | `TypeError`                              | general; `cls` is a type, a tuple of types or a union; `bool` is an `int` here (documented)            |
| Type      | `validate_int(value, *, name)`                                                       | `TypeError`                              | the only shortcut with its own rule: `bool` rejected; `numpy.int64` rejected (strict)                  |
| Type      | `validate_number(value, *, name)`                                                    | `TypeError`                              | `numbers.Real`, `bool` rejected; NumPy scalars, `Fraction` accepted; `Decimal` rejected (not a `Real`) |
| Bound     | `validate_ge`, `validate_gt`, `validate_le`, `validate_lt` `(value, bound, *, name)` | `TypeError` (incomparable), `ValueError` | duck-typed; the sign checks are `validate_ge(x, 0)` ...                                                |
| Range     | `validate_range(value, low, high, *, low_inclusive=True, high_inclusive=True, name)` | `TypeError`, `ValueError`                | one function for open, closed and half-open ranges; closed by default (the form most call sites use)   |
| Finite    | `validate_finite(value, *, name)`                                                    | `TypeError` (not a number), `ValueError` | calls `validate_number` first: `bool`, `None`, `str`, arrays give a named `TypeError`                  |
| Container | `validate_not_empty(value, *, name)`                                                 | `TypeError` (no `len`), `ValueError`     | `len(value) == 0`; an iterator or a generator has no `len` and is a `TypeError`, not consumed          |
| Container | `validate_in(value, valid, *, name)`                                                 | `ValueError`                             | an unhashable value is not in a `set`: `ValueError`, not a raw `TypeError`; listing bounded            |

There are no type shortcuts such as `validate_str` or `validate_list` (use `validate_isinstance(value,
X)`), no sign shortcuts such as `validate_positive` (use `validate_gt(value, 0)`), and no separate
`open`, `closed` and `half_open` range functions (use `validate_range`). The synonyms are the part of
a package that grows one name at a time; without them the list of names can be read in ten seconds and
the rule to add one is written down. Keeping `validate_positive` and `validate_non_negative` only would
need an explanation for the missing `negative`.

The rule to add a function (written in the module docstring and in the guide): _a
validator exists only if it states a constraint that `isinstance` or one comparison cannot state
without a special case_. `validate_int` exists for the `bool` case; `validate_number` for the
`numbers.Real` plus `bool` case; `validate_finite` for the overflow of `math.isfinite` and the
NaN. `validate_str` does not: `validate_isinstance(x, str)` is the same line.

### Modules

Three modules, named by what the function checks:

| Module          | Contains                                                                | Why                                                      |
| --------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- |
| `type.py`       | `_type_error`, `validate_isinstance`, `validate_int`, `validate_number` | what a value **is**                                      |
| `comparison.py` | `_compare`, `validate_ge/gt/le/lt`, `validate_range`, `validate_finite` | what a value **compares** to (finite = `-inf < x < inf`) |
| `container.py`  | `validate_not_empty`, `validate_in`                                     | what a value **holds** or belongs to                     |

`__init__.py` re-exports the 14 names; `__all__` is alphabetical. `docs/docs/refs/validation.md` lists
the three submodules.

## Dependents

The messages have one format, so the tests of the callers that match a message match this format.

| Package                                                                      | Use                                                                                                                                                                                 |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `utils` (`bloom_filter`, `batching`, `text_diff`, `format`, `secret`, `lru`) | `validate_gt(x, 0)`, `validate_ge(x, 0)`.                                                                                                                                           |
| `equality.config`, `equality.tolerance`, `summary.config`                    | The same, and `Tolerance` uses `validate_number`, a bound and `validate_finite`.                                                                                                    |
| `identifier` (`objectid`, `ulid`, `uuid7`, `snowflake`, `checksummed`)       | `validate_range(...)`, `validate_isinstance(x, str)`, `validate_gt(x, 0)`. `_validate_sep` has its own checks (the Crockford rule) and calls `validate_isinstance(sep, str)`.       |
| `nested.keys`, `nested.mapping`                                              | `validate_isinstance(x, Mapping)`; `validate_in(on_duplicate, ("raise", "first", "last", "prefix"), name="on_duplicate")` (the valid values are a tuple, so the message is stable). |
| `random.seed.validate_seed`                                                  | `validate_int(seed, name="seed")` then `validate_range(seed, 0, MAX_SEED, high_inclusive=False, name="seed")`; it lives in `coola.random`, which owns the 2**63 bound.              |
| `hashing.engine.validate_hash_length`                                        | `validate_int` and `validate_range(length, 2, 128)`; the parity rule (`length % 2`) is local, and the hot-path fast test comes first (it is measured).                              |
| `rounding.common.validate_arguments`                                         | `validate_in(mode, MODES, name="mode")`, `validate_int(decimals)`, `validate_ge(decimals, 0)`.                                                                                      |
| `registry.registry`, `registry.type` (`_validate_key`)                       | Not migrated: the message `Expected a type as key` is part of its own contract and tests. A candidate for a later item.                                                             |

No dependent imports anything private from `coola.validation`.

## Design choices and trade-offs

**`numpy.int64` and `numpy.float32` are not ints and floats for the type validators, but numbers for
`validate_number`.** A type annotation `int` rejects `numpy.int64` for a type checker, `validate_int`
follows the annotation, and a caller that accepts NumPy scalars writes `validate_isinstance(x, (int,
np.integer))` or uses `validate_number`. `numbers.Integral` for `validate_int` would accept NumPy ints
(still rejecting `bool` by the explicit test) with a one-line change, but it makes the strictness
depend on the registration of the ABCs, and an `int` argument that is later used as `range(n)` or a
shift accepts `numpy.int64` anyway. The strict rule can be revisited if a user reports a real case.

**"Type then bound" is two calls.** `validate_int(x, min=0)` would remove 8 lines at the call sites and
add options to the most used function, with a second message path. Two calls are explicit, and the
order of the errors (a `TypeError` for `"a"` before a `ValueError` for `-1`) is visible. Revisit if a
call site needs three constraints on one argument.

**An empty range raises.** `validate_range(x, 1, 0)` cannot succeed; it is a programming error, and
blaming the value would send the reader to the wrong argument. The `ValueError` about the bounds costs
one comparison.

**`validate_in` with an unhashable value is a `ValueError`.** From the point of view of the caller, `[]`
is not one of the valid modes; the fact that the container is a `set` is an implementation detail. A
`TypeError` would be right only if `valid` were not a container, and that case still raises (the bug of
the caller). A `valid` that is a string with a non-string value (`1 in "abc"` raises `TypeError`) is
reported as a value error, and the message shows both.

**The project-wide rule lives in this package's docstring, the guide and `CLAUDE.md`:** "Validate an
argument with `coola.validation`; do not write `raise ValueError` for a type, range or membership check,
and do not add a validator that is one `isinstance` or one comparison." Without the rule, each package
grows private copies with their own messages.

**No decorator or `Annotated` metadata.** A schema layer (pydantic, attrs validators, beartype) does it
better, and none of the call sites needs it. The package is a set of functions that a constructor
calls.

## Not covered

- **Aggregated errors** (all the invalid fields of a config in one exception). The call sites raise
  on the first error, as the standard library does.
- **Custom exception classes** and error codes. The two builtin types are the contract.
- **Validation of nested structures and of array shapes or dtypes.** `validate_isinstance(x,
np.ndarray)` works; a dtype check belongs to the package that owns the array library (lazy import).
- **Coercion** (`ensure_int`, `to_float`). Validators never convert (property "silent on success").
- **`registry._validate_key`**: local to `registry`.
- **A `validate_callable`, `validate_path`, `validate_enum`, ...**: each is
  `validate_isinstance(x, Callable)` or `validate_in(x, tuple(Enum))`; they are added only if they
  carry a rule of their own (see the rule above).
