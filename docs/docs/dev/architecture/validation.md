# Architecture of `coola.validation`

:book: This page describes the design of the `coola.validation` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../../uguide/validation.md#properties); the API is in the [reference](../../refs/validation.md).

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
[user guide](../../uguide/validation.md#properties) for the properties of the package.

## Design constraints

1. **Simple and easy to maintain.** The package is 804 lines for 26 functions, 13 of which are
   one-line wrappers (nine type checks are `validate_isinstance(value, X)`, four sign checks are
   `validate_ge/gt/le/lt(value, 0)`) and three are one range with two flags. Eight of the 26
   functions have no caller in `src`. Each wrapper costs a docstring, an export, a doc entry and tests. The target
   keeps a function only if it adds behaviour that the caller would otherwise have to write.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period. The release notes list the removed names and the call-site rewrite.
3. **Core package.** `coola.validation` imports only the standard library (import-linter contract
   "Core packages must not import feature packages"). It imports NumPy or any optional library
   **never**; NumPy scalars are supported through the `numbers` ABCs they register with.
4. **Cheap.** Validators run in constructors and in the entry points of `hash_object`,
   `round_values`, `convert_to_jsonable`, ... A call costs 70 to 240 ns before. The
   target must not be slower on the success path.
5. **One home for argument checks.** The checks that are re-implemented next to their caller
   (`validate_seed`, `validate_hash_length`, `rounding.common.validate_arguments`,
   `_validate_on_duplicate`, `_validate_sep`, `_validate_key`) use the validators of this package
   where they can, so the same mistake gives the same message everywhere.

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

`_compare` replaces `_check` **and** `_check_range`. The four bound checks, the range and the
finiteness check call it with a `lambda`/`operator.partial` test; the closure costs about 40 ns
, below the margin of the gate. Both helpers stay private in `coola.validation`: they are
an implementation detail, and a package that wants the same format writes a call to a validator.

### The functions (target API)

`from coola.validation import ...` (14 names, down from 26):

| Family    | Function                                                                             | Raises                                   | Notes                                                                                                                |
| --------- | ------------------------------------------------------------------------------------ | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Type      | `validate_isinstance(value, cls, *, name)`                                           | `TypeError`                              | general; `cls` is a type, a tuple of types or a union; `bool` is an `int` here (documented)                          |
| Type      | `validate_int(value, *, name)`                                                       | `TypeError`                              | the only shortcut with its own rule: `bool` rejected; `numpy.int64` rejected (strict, decision D2)                   |
| Type      | `validate_number(value, *, name)`                                                    | `TypeError`                              | `numbers.Real`, `bool` rejected; NumPy scalars, `Fraction` accepted; `Decimal` rejected (not a `Real`)               |
| Bound     | `validate_ge`, `validate_gt`, `validate_le`, `validate_lt` `(value, bound, *, name)` | `TypeError` (incomparable), `ValueError` | duck-typed; the sign checks are `validate_ge(x, 0)` ...                                                              |
| Range     | `validate_range(value, low, high, *, low_inclusive=True, high_inclusive=True, name)` | `TypeError`, `ValueError`                | replaces `open`, `closed` and `half_open`; closed by default (the form all 10 call sites of the range functions use) |
| Finite    | `validate_finite(value, *, name)`                                                    | `TypeError` (not a number), `ValueError` | calls `validate_number` first: `bool`, `None`, `str`, arrays give a named `TypeError`                                |
| Container | `validate_not_empty(value, *, name)`                                                 | `TypeError` (no `len`), `ValueError`     | `len(value) == 0`; an iterator or a generator has no `len` and is a `TypeError`, not consumed                        |
| Container | `validate_in(value, valid, *, name)`                                                 | `ValueError`                             | an unhashable value is not in a `set`: `ValueError`, not a raw `TypeError`; listing bounded                          |

Removed: `validate_bool`, `validate_float`, `validate_str`, `validate_list`, `validate_tuple`,
`validate_dict`, `validate_mapping`, `validate_set`, `validate_frozenset` (replaced by
`validate_isinstance(value, X)`, ); `validate_positive`, `validate_non_negative`,
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

### Modules

Three modules instead of four, named by what the function checks:

| Module          | Contains                                                                | Why                                                      |
| --------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- |
| `type.py`       | `_type_error`, `validate_isinstance`, `validate_int`, `validate_number` | what a value **is**                                      |
| `comparison.py` | `_compare`, `validate_ge/gt/le/lt`, `validate_range`, `validate_finite` | what a value **compares** to (finite = `-inf < x < inf`) |
| `container.py`  | `validate_not_empty`, `validate_in`                                     | what a value **holds** or belongs to                     |

`numeric.py` is deleted. `__init__.py` re-exports the 14 names; the order of `__all__` stays
alphabetical. `docs/docs/refs/validation.md` already lists three submodules and is unchanged.

## Dependents

The migration of the call sites is mechanical and each rewrite is a pure rename. The messages change
(the unified format), so the tests of the callers that match a message change with them.

| Package                                                                      | Change                                                                                                                                                                                                                                                                                                 |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `utils` (`bloom_filter`, `batching`, `text_diff`, `format`, `secret`, `lru`) | `validate_positive(x)` becomes `validate_gt(x, 0)`, `validate_non_negative(x)` becomes `validate_ge(x, 0)`. Nothing else.                                                                                                                                                                              |
| `equality.config`, `equality.tolerance`, `summary.config`                    | Same rename. `Tolerance` keeps `validate_number`, bound, `validate_finite`.                                                                                                                                                                                                                            |
| `identifier` (`objectid`, `ulid`, `uuid7`, `snowflake`, `checksummed`)       | `validate_closed_range(...)` becomes `validate_range(...)`; `validate_str(x)` becomes `validate_isinstance(x, str)`; `validate_positive(x)` becomes `validate_gt(x, 0)`. `_validate_sep` keeps its own checks (the Crockford rule) and calls `validate_isinstance(sep, str)`.                          |
| `nested.keys`, `nested.mapping`                                              | `validate_mapping(x)` becomes `validate_isinstance(x, Mapping)`. `_validate_on_duplicate` is deleted: `validate_in(on_duplicate, ("raise", "first", "last", "prefix"), name="on_duplicate")` (the valid values become a tuple, so the message is stable). `nested.polars`, `nested.flat`: rename only. |
| `random.seed.validate_seed`                                                  | Becomes `validate_int(seed, name="seed")` then `validate_range(seed, 0, MAX_SEED, high_inclusive=False, name="seed")`, behind the same public function (it keeps its place in `coola.random`, which owns the 2**63 bound).                                                                             |
| `hashing.engine.validate_hash_length`                                        | `validate_int` + `validate_range(length, 2, 128)`; the parity rule (`length % 2`) stays local. The hot-path fast test stays first, as before (it is measured).                                                                                                                                         |
| `rounding.common.validate_arguments`                                         | `validate_in(mode, MODES, name="mode")`, `validate_int(decimals)`, `validate_ge(decimals, 0)`: the private `bool` test and the hand-made messages are deleted.                                                                                                                                         |
| `registry.registry`, `registry.type` (`_validate_key`)                       | Not changed: `registry` is a core package and a `validate_isinstance(key, type)` call would need no more code, but the message `Expected a type as key` is part of its own contract and tests. Candidate for a later item.                                                                             |

No dependent imports anything private from `coola.validation`.

## Decision record

All seven decisions are settled: D1 to D6 follow the recommendation, D7 is "no". The design above
is the decided design; the text of each decision is kept as the reason.

**D1. Delete the synonyms?** _Recommended: yes._ Thirteen of the 26 functions are a one-line wrapper
and eight have no caller. They are the part of the package that grows (the range functions
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
remove 8 lines at the call sites and add options to the most used function, with a second
message path. Two calls are explicit, and the order of the errors (a `TypeError` for `"a"` before a
`ValueError` for `-1`) is visible. Revisit if a call site needs three constraints on one argument.

Decision: **adopted**.

**D4. Raise on an empty range?** _Recommended: yes._ `validate_range(x, 1, 0)` cannot succeed;
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
`isinstance` or one comparison." shows what happens without the rule: six private copies with
their own messages.

Decision: **adopted**.

**D7. Make `validate_*` available as a decorator or an `Annotated` metadata?** _Recommended: no._ A
schema layer (pydantic, attrs validators, beartype) does it better, and none of the call sites needs
it. The package stays a set of functions that a constructor calls.

Decision: **no**, the package stays a set of functions.

## Not covered

- **Aggregated errors** (all the invalid fields of a config in one exception). The call sites raise
  on the first error, as the standard library does.
- **Custom exception classes** and error codes. The two builtin types are the contract.
- **Validation of nested structures and of array shapes or dtypes.** `validate_isinstance(x,
np.ndarray)` works; a dtype check belongs to the package that owns the array library (lazy import).
- **Coercion** (`ensure_int`, `to_float`). Validators never convert (property "silent on success").
- **`registry._validate_key`**: kept local, a candidate for a later item.
- **A `validate_callable`, `validate_path`, `validate_enum`, ...**: each is
  `validate_isinstance(x, Callable)` or `validate_in(x, tuple(Enum))`; they are added only if they
  carry a rule of their own (rule of).
