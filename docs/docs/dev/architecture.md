# Architecture and Design

This document describes the internal architecture and design principles of `coola`.

## Overview

`coola` is designed around a flexible, extensible comparison framework that can handle various data
types through a plugin-like architecture. The core design follows these principles:

1. **Separation of concerns**: Comparison logic is separated from data type handling
2. **Extensibility**: New data types can be added without modifying core code
3. **Type safety**: Strong type checking to prevent subtle bugs
4. **Composability**: Complex comparisons are built from simpler ones

## Core Components

### 1. Comparison Functions

The main entry points for users:

- **`objects_are_equal`**: Checks exact equality
- **`objects_are_allclose`**: Checks equality within tolerance

These functions provide a simple interface while delegating to the internal comparison system.

### 2. Testers

A tester implements the comparison logic for one type (or family of types). Testers live in
`coola.equality.tester`.

#### `BaseEqualityTester`

Abstract base class defining the tester interface:

```python
class BaseEqualityTester(ABC, Generic[T]):
    @abstractmethod
    def objects_are_equal(self, actual: T, expected: T, ctx: Engine) -> bool | Iterator[tuple]:
        """A verdict, or the child pairs (actual, expected, path part) that must all be equal."""
```

#### `BaseEqualityTester` and the engine

The Python testers use the engine (`coola.equality.engine.Engine`). The engine owns the invariants of
a comparison: identity, strict type, depth, cycles and the location of the first difference. A tester
that derives from `BaseEqualityTester` never recurses: its `objects_are_equal(actual, expected,
ctx)` receives two objects of the same type and the engine, and returns a verdict (`True`, or
`ctx.fail(reason, *path)`, which returns `False`) or the child pairs `(actual, expected, path part)`
that must all be equal, produced lazily. The engine keeps the child iterators on an explicit stack, so
there is no `RecursionError` whatever the nesting, and a comparison stops at the first difference.

```python
class PointTester(BaseEqualityTester[Point]):
    def objects_are_equal(self, actual, expected, ctx):
        return ((actual.coords, expected.coords, AttributeName("coords")),)
```

The testers of NumPy, PyTorch, JAX and xarray use the engine too: an array tester checks the data
type and the shape (and the device for a tensor), then compares the elements with the shared kernel;
the location of a difference is the index of the first element that differs, looked up only when the
difference is explained. #### Built-in Testers

- **`DefaultEqualityTester`**: Fallback for any type (`==` on two objects of the same type)
- **`MappingEqualityTester`**, **`OrderedMappingEqualityTester`** (also checks the key order, used
  for `OrderedDict`), **`SequenceEqualityTester`** and **`SetEqualityTester`**: Return the children of
  mappings, sequences and sets
- **`DataclassEqualityTester`**, **`ObjectAttributesEqualityTester`** and
  **`PydanticModelEqualityTester`**: Return the fields or the attributes as children
- **`ScalarEqualityTester`**, **`DifferenceCloseEqualityTester`**, **`EqualEqualityTester`**,
  **`EqualNanEqualityTester`**, **`TolerantEqualEqualityTester`**: Numbers and objects with an
  `equal` method
- Array testers for NumPy, PyTorch, xarray and JAX (for example `NumpyArrayEqualityTester`,
  `TorchTensorEqualityTester`)
- **`TableEqualityTester`**: pandas, polars and PyArrow tables, with one adapter per library

### 3. Registry

#### `EqualityTesterRegistry`

The registry maps types to testers. It is a `TypeRegistry` (`coola.registry`):

- `find_equality_tester(data_type)` returns the tester for the most specific registered type,
  walking the Method Resolution Order (MRO); `object` is registered with `DefaultEqualityTester`,
  so a tester is always found
- `objects_are_equal(actual, expected, config)` finds the tester for `type(actual)` and delegates (a tester that uses the engine runs a nested `Engine`)
  to it
- Lookup results are cached per type, and the cache is cleared whenever the registry changes

`get_default_registry()` returns the global registry. Testers for optional libraries are only
registered when the library is installed. `register_equality_testers(mapping, *, exist_ok=False)`
adds testers to it. A test that registers a tester should not leak it: use
`coola.utils.singleton.temporary_default_registries()` (see the testing guide), or work on
`get_default_registry().copy()`.

### 4. Configuration

#### `EqualityConfig`

A frozen dataclass that carries the comparison settings. It can be shared between comparisons and
threads; the state of one comparison (the cache of testers, the first failure) lives in the `Engine`:

```python
@dataclass(frozen=True, slots=True)
class EqualityConfig:
    registry: EqualityTesterRegistry  # defaults to the default registry
    tolerance: Tolerance = Tolerance()  # rtol, atol
    equal_nan: bool = False
    max_depth: int = 100
```

The tester reads the settings from the engine (`ctx.atol`, `ctx.rtol`, `ctx.equal_nan`).

### 5. Comparison Results

`compare` returns a `ComparisonResult`, and `assert_objects_equal` / `assert_objects_allclose`
raise an `AssertionError` with a description of the difference.

## Data Flow

Here's how a comparison flows through the system:

```
User calls objects_are_equal(obj1, obj2)
    ↓
Creates EqualityConfig with settings (and the registry)
    ↓
Runs Engine(config).run(obj1, obj2)
    ↓
For each pair: identity, strict type, tester lookup (cached per call, MRO lookup)
    ↓
Calls tester.objects_are_equal(obj1, obj2, engine): a verdict, or the child pairs
(pushed on the explicit stack, depth first)
    ↓
Returns boolean result
```

## Design Patterns

### 1. Strategy Pattern

Testers implement different comparison strategies for different types, allowing the algorithm to
vary independently from the clients that use it.

### 2. Template Method

Many testers follow a template:

1. Check types match
2. Check metadata (shape, dtype, etc.)
3. Check values
4. Optionally show differences

### 3. Registry Pattern

The tester registry allows runtime type-to-tester mapping, enabling extensibility.

### 4. Visitor Pattern

The recursive nature of comparison through nested structures follows a visitor-like pattern.

## Extension Points

### Adding Support for New Types

To add support for a custom type:

1. **Implement a Tester:**

   ```python
   from coola.equality.config import EqualityConfig
   from coola.equality.tester import BaseEqualityTester


   class MyTypeEqualityTester(BaseEqualityTester[MyType]):
       def equal(self, other: object) -> bool:
           return type(other) is type(self)

       def objects_are_equal(self, actual: MyType, expected: object, config: EqualityConfig) -> bool:
           # Type check
           if type(actual) is not type(expected):
               return False

           # Custom comparison logic
           return actual.compare_to(expected)
   ```

2. **Register the Tester:**

   ```python
   from coola.equality.tester import register_equality_testers

   register_equality_testers({MyType: MyTypeEqualityTester()})
   ```

   To avoid modifying the global registry, create a `EqualityTesterRegistry`, register the tester
   on it and pass it with `registry=`.

3. **Use it:**

   ```python
   from coola.equality import objects_are_equal

   objects_are_equal(obj1, obj2)
   ```

See the [extending guide](../uguide/extending.md) for more details.

### Checklist: writing a tester

- Return a verdict (`ctx.fail(reason)` for a difference) or the child pairs `(actual, expected,
path part)`; never recurse: the engine checks the identity, the type, the depth and the cycles
- Use `AttributeName` for an attribute and `ctx.fail(reason, *path)` for the location inside a leaf
  (the index in an array)
- Build the reason lazily (a callable), so `objects_are_equal` formats nothing
- Honour `ctx.equal_nan` and the tolerance (`ctx.rtol`, `ctx.atol`: `|a - e| <= atol + rtol *
max(|a|, |e|)`), or document that your tester ignores them

## Type System

### Strict Type Checking

`coola` enforces strict type checking:

- `1` (int) ≠ `1.0` (float) ≠ `True` (bool)
- `list` ≠ `tuple`
- `dict` ≠ `OrderedDict`
- two `OrderedDict`s with the same items in a different order are different (the order of the keys of
  a `dict` is ignored)

This prevents subtle bugs from type coercion.

### Type Hierarchy Support

Through MRO-based lookup, `coola` supports inheritance:

- A tester for `Sequence` applies to `list`, `tuple`, etc.
- More specific testers override general ones
- Custom subclasses inherit parent testers

## Performance Considerations

### Early Exit

Testers check fast properties first:

1. Type check (very fast)
2. Metadata checks (fast: shape, dtype, device)
3. Value comparison (potentially slow)

### Lazy Evaluation

Comparisons short-circuit on first difference when possible.

### Caching

The registry caches tester lookups by type for performance.

### Recursive Transformation

`recursive_apply` uses one engine (`coola.recursive.engine`) and one extension point. A transformer
only describes the shape of a type: `expand(data)` returns `None` for a leaf, or the children and
the function that rebuilds the container from the transformed children. The engine owns the
invariants: the transformer lookup (by MRO, cached for the call), the call of `func` on the leaves,
the cycle check and the depth (an explicit stack, so there is no `RecursionError`). This is the same
split as the equality engine, where a tester returns the child pairs.

### Nested

`coola.nested` has no traversal of its own: a function that goes inside the values uses the engine of
`recursive` (or the path iterator `dfs_items` of `iterator`) and describes one type, and a function
that works on the top level only is a plain loop. The key functions (`remove_keys_*`,
`add_prefix_suffix_to_keys`) use one private transformer for the mappings, registered in a copy of
the default registry; `convert_to_jsonable` adds a transformer for the sets (a sorted list) and one
for the dataclasses (a `dict`); `to_flat_dict` joins the paths of `dfs_items`; `from_flat_dict`
builds its result with a private `dict` subclass and one final `recursive_apply`. The containers,
the cycle check and the depth are therefore the ones of the other packages.

### Summary

`summarize` uses one engine (`coola.summary.engine`), one extension point and one option object
(`SummaryConfig`). A summarizer only describes one type: `summarize(data, config, *, compact)`
returns a `str` for a leaf, or a `Node` (a header, the lazy `(label, child)` pairs and their number)
for a container. The engine owns the invariants: the summarizer lookup (by MRO), the display depth,
`max_items` and `max_characters`, the cycle check, the layout, and the errors (a summarizer that
raises is replaced by a placeholder). It walks the nodes with an explicit stack, so there is no
`RecursionError`. A container at the display depth is shown on one line by `CompactRepr`, whose work
is bounded by `max_items`, so the cost depends on the size of the summary and not on the size of the
data. This is the same split as the equality and the recursive engines.

### Depth

The engine does not recurse (see [Design of `coola.equality`](#design-of-coolaequality)), so a deep
structure cannot overflow the interpreter stack. `EqualityConfig.max_depth` (default 100) bounds the
work on a pathological structure and raises `MaxDepthExceededError`, a plain `Exception`.

## Error Handling

### Graceful Degradation

When a specific tester is not available, `coola` falls back to:

1. More general tester (via MRO)
2. `DefaultEqualityTester` (registered for `object`), which uses `==`

### Informative Messages

When the comparison explains the difference (`compare`, `assert_*`), testers report:

- What objects differ
- Where in the structure the difference is
- The actual values that differ

## Testing Strategy

The `coola` codebase uses:

1. **Unit tests**: Test individual testers in isolation
2. **Integration tests**: Test complete comparison workflows
3. **Property-based tests**: Test invariants (reflexivity, symmetry of the exact comparison, and
   monotonicity of `*_allclose`: a larger `atol` or `rtol` never turns a `True` into `False`). The
   unit tests also walk the default registry: every tester of a built-in type must be reflexive on
   a sample of its type (`NATIVE_SAMPLES` in `tests/unit/equality/tester/test_interface.py`)
4. **Cross-library tests**: Test integration with PyTorch, NumPy, etc.

## Dependencies

### Core Dependencies

- Python 3.11+: Core language features

### Optional Dependencies

- **torch**: PyTorch tensor support
- **numpy**: NumPy array support
- **pandas**: DataFrame support
- **polars**: Polars DataFrame support
- **xarray**: xarray support
- **jax**: JAX array support
- **pyarrow**: PyArrow table support

Each optional dependency is only imported when used (lazy loading): `import coola` and the
import of its subpackages do not import any of them. The `__init__.py` files expose the names that
need a library with `lazy_exports`, and the default registries load the entries of a library the
first time they look for one of its types (`register_lazy`). See
[Lazy imports](add_backend.md#lazy-imports).

## Module Organization

```
coola/
├── equality/             # Equality and tolerance comparison
│   ├── interface.py      # objects_are_equal, objects_are_allclose
│   ├── result.py         # compare, assert_objects_equal, ...
│   ├── config.py         # EqualityConfig
│   ├── tester/           # Type-specific testers and the registry
│   └── handler/          # Helpers: numeric kernels, formatting, path parts
├── registry/             # Generic registries and type dispatch
├── summary/, hashing/, recursive/, iterator/, nested/, random/, reducer/
├── io/, factory/, identifier/, display/, validation/, testing/
└── utils/                # Utility functions
```

## Design of `coola.equality`

The package is small on purpose: one engine, one extension point, one numeric rule and one table
algorithm.

### Constraints

1. **Performance is a requirement.** Both the passing and the failing path are measured by the
   benchmarks; a simpler design that is slower is rejected.
2. **Early exit.** Two objects are different as soon as one difference is found: nothing after it is
   visited, converted or formatted.
3. **First difference only, on demand.** `objects_are_equal` and `objects_are_allclose` build no
   message and no path; only `compare` and `assert_*` explain, and only the first difference.
4. **Equality drives hashing, not the reverse.** `hash_object` is aligned with equality, never the
   other way round.

### Architecture

```
objects_are_equal / objects_are_allclose / compare / assert_objects_equal / assert_objects_allclose
        |   one EqualityConfig (frozen) and one Engine per call
        v
Engine.run(actual, expected)    explicit stack of child iterators; one loop
        |   identity -> type -> tester lookup (cached per call) -> verdict or children
        v
Tester.objects_are_equal(actual, expected, ctx) -> bool | Iterator[(actual, expected, path part)]
        |   leaves: numbers, arrays, scalars       containers: sequence, mapping, set, object, table
        v
numeric kernel (NumPy, torch, JAX)      adapters: pandas, polars, pyarrow -> table
```

Four rules follow from it:

1. **One extension point: the tester.** A tester is a class with one method.
2. **The engine owns the invariants**: identity, strict type, depth, cycles, path and first failure.
   A tester cannot forget them because it never recurses.
3. **A container decomposes, it does not recurse**: it returns the child pairs that must all be equal,
   lazily and in the order the first difference is wanted.
4. **coola owns the semantics, the libraries own the structure**: every number goes through the
   kernel; a library says whether two frames have the same columns, never whether `1` is close to `2`.

### The engine

The engine is one loop over a stack of child iterators (depth first, so the first difference in
iteration order is the one reported). For each pair:

1. `actual is expected`: equal.
2. `type(actual) is not type(expected)`: different.
3. The tester is looked up in a per-call `dict[type, tester]` filled from the registry the first time
   a type is met, so the registry lookup is paid once per type. A registration made during a
   comparison is seen by the next call.
4. The tester is called. A `bool` is the verdict. An iterator is a container: if its pair is already
   on the stack the pair is equal (cycle); if the stack is `max_depth` deep, `MaxDepthExceededError`
   is raised; otherwise it is pushed and the loop continues with its first child.
5. On `False`, the path is built once from the path parts of the stack, the part of the failing pair
   and the parts given to `ctx.fail`.

Consequences: the stack use is constant, `RecursionError` cannot happen, and explaining costs nothing
in the loop. `ctx.fail(reason, *path)` returns `False`; `reason` can be a callable that is called only
when the difference is explained.

### Reporting

There is one record, the first failure `(path, reason)`: no list of reports and no message per parent
level. Values in a reason go through one bounded formatter, so a failure in a million-element array
formats a few values, not the array. A path part is a key or a position (`['a']`, `[3]`),
`AttributeName("x")` (`.x`) or `Element(2)` (`{2}`). An array reports its index. A table is a leaf
and names only the first differing column; the user finds the row with the library (for example
`DataFrame.compare`).

### The numeric kernel

NumPy, torch and JAX share one formula:

```
close = (a == b) | (finite & (|a - b| <= atol + rtol * max(|a|, |b|))) | (isnan(a) & isnan(b) if equal_nan)
```

| Dtype                                | Preparation                           | Rule                             |
| ------------------------------------ | ------------------------------------- | -------------------------------- |
| `bool`                               | none                                  | exact                            |
| signed and unsigned integers         | exact difference in integers          | kernel, tolerance in `float64`   |
| `float16`, `bfloat16`, `float32`     | cast to `float64`                     | kernel                           |
| `float64`, `complex128`              | none (`complex64` is cast up)         | kernel                           |
| `timedelta64`                        | integer nanoseconds; `NaT` mask apart | kernel; `NaT` by `equal_nan`     |
| `datetime64`, `str`, `bytes`, `void` | none                                  | exact; `NaT` by `equal_nan`      |
| `object`                             | none                                  | cells are children of the engine |

Python scalars use the same formula; `Decimal`, `Fraction` and `timedelta` keep a difference-based
rule.

### Tables

pandas, polars and PyArrow share one algorithm in `TableEqualityTester`, which takes an adapter per
library: check the structure (shape, names, dtypes, index), then compare the columns lazily, one by
one, each by its kind, and stop at the first column that differs. A `Series` or an `Array` is a
one-column table.

| Kind       | Dtypes                                                       | Comparison                                                      |
| ---------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| `numeric`  | integers, floats, nullable and Arrow numerics                | null masks equal, then the kernel                               |
| `duration` | `timedelta64`, polars `Duration`, Arrow `duration`           | kernel on integer nanoseconds                                   |
| `exact`    | booleans, strings, categories, instants, nested and `object` | the library's `equals`; never sees the tolerance or `equal_nan` |

Nested and `object` columns are the one place where the library decides. `attrs`, `flags` and Arrow
schema metadata are ignored. Adding a library means writing one adapter of about 40 lines.

### Module layout

```
coola/equality/
  interface.py   the five functions
  config.py      Tolerance, EqualityConfig
  engine.py      Engine, Element
  result.py      ComparisonResult
  tester/        base, registry, one module per family (default, scalar, collection, difference,
                 dataclass, object, equal, tolerant), per library (numpy, torch, jax, xarray,
                 pydantic), and table.py with the adapters pandas.py, polars.py, pyarrow.py
  _helpers/      private helpers (formatting, numeric kernel, comparison keys)
```

### Decisions

| Decision                                                               | Reason                                                |
| ---------------------------------------------------------------------- | ----------------------------------------------------- |
| Iterative engine, not recursion                                        | constant stack, one place for depth, cycles and path  |
| One tester method returns a verdict or an iterator                     | one extension point                                   |
| No `show_difference` and no logger; `compare` and `assert_*` explain   | no cost on the fast path, thread safe                 |
| `bool` and every `Enum` are exact                                      | they are labels, not quantities                       |
| Masked arrays compare the data under the mask, the mask and fill value | no hidden difference                                  |
| Small floats are cast to `float64`                                     | no overflow of the tolerance at the edge of the range |
| Integer difference is exact                                            | no rounding of large integers                         |
| Tables are one leaf, reported by column                                | simple and fast; the library finds the row            |
| Time resolution is part of the dtype                                   | as in NumPy                                           |
| Library testers are public                                             | users can compose, subclass or register them          |
| `max_depth=100` and `MaxDepthExceededError` are kept                   | bound the work on pathological structures             |

## Design of `coola.factory`

The package builds an object from a fully qualified name. It is small on purpose and has no state:
it keeps no registry, cache or singleton, and it imports the standard library and
`coola.utils.imports` only. A configuration names the code to run, so every change is judged first
by what the data can make the process do. The three entry points (`factory`, `resolve_object` and
`is_object_config`) are compositions of functions that each own one invariant: `import_object` is the
only place that imports a target (allow-list checked before the import, one `ImportError` for a
missing target), and `instantiate_object` is the only place that calls one (abstract classes
refused, the `_init_` rule, the target's keywords never shadowed by a parameter of the API). The
allow-list is a filter on a path and always comes from the caller's code, never from the data. See
the [user guide](../uguide/factory.md#properties) for the properties of the package.

## Design of `coola.io`

The package reads and writes one value to one file, and a write either happens completely or leaves
the target as it was. The base classes own the invariants and a format only describes how to turn a
value into a file and back. `BaseLoader.load` is a template method: it converts the path (`str`,
`file://` URI or `os.PathLike`) and calls the `_load(Path)` of the format, so a format cannot forget
the conversion. `BaseFileSaver.save` writes to a hidden temporary file next to the target and
commits it with one atomic operation (`os.link` for `exist_ok=False`, `os.replace` for
`exist_ok=True`), then removes the temporary file, so a reader sees the old file or the new one and
two concurrent saves never mix. There is no lock, no lock file and no thread: the commit already
decides every race, and a caller that needs mutual exclusion around a read-modify-write uses its own
lock. `equal()`, `repr` and `str` are written once in the base classes from `_display_fields()`; a
format that has no configuration defines nothing. `TextSaver` accepts a `str` only. Optional
libraries (torch) are imported lazily. See the [user guide](../uguide/io.md#properties) for the
properties of the package.

## Design of `coola.identifier`

The package has **two families and one clock rule**. The content-derived identifiers
(`content.py`) are a function of the data through `coola.hashing`; the time-ordered identifiers
(`ulid.py`, `uuid7.py`, `objectid.py`, `snowflake.py`) are a **format** (a bit layout and an encoding)
on top of **one engine**, `TickSequence` (`sequence.py`). A format module has no lock, no clock and no
fork code: it asks the engine for a `(tick, counter)` pair and packs it. The engine has three cases and
none of them raises or waits: a new tick restarts the counter (random, or `0` for Snowflake whose
uniqueness comes from its `worker_id`), the same tick or a clock that went back increments it, and an
exhausted counter borrows the next tick, so the timestamp of an identifier can run ahead of the clock
until the clock catches up. The state is per object and protected by a lock, and
`coola.utils.fork.track_fork` resets it in a forked child; a `SnowflakeIdGenerator`, whose `worker_id`
the child cannot choose, raises there instead. `crockford.py` is the one Crockford Base32 encoder and
decoder (ULID and the checksummed identifiers), and the range and type checks come from
`coola.validation`. The extractors are the strict inverse of the generators: they accept the
canonical form only. See the [user guide](../uguide/identifier.md#properties) for the properties of
the package.

## Design of `coola.validation`

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

## Design of `coola.hashing`

The package has **one engine, one extension point and one encoding rule**. A hasher only describes
the content of a type: `encode(data)` returns the bytes of a leaf (`tagged(tag, payload)`) or a
`Node(tag, children, ordered, group)`. The engine (`coola.hashing.engine`) owns everything else: the
dispatch by type (once per type and per call), the digests, the order rule, the cycles, the shared
containers, the depth and the placeholder of `ignore_unhashable`. This is the same split as
`equality` (the engine owns identity, type, depth and cycles; a tester returns the child pairs) and
`recursive` (a transformer returns the children).

```
hash_object(data, *, registry=None, length=64, ignore_unhashable=False)
        |   validates length; one registry; one call = one engine run
        v
digest_of(data, registry, ...)       explicit stack of frames; per-call type -> hasher cache and memo
        |-- hasher   = registry.find_hasher(type(data))     MRO, fallbacks, Enum rule
        |-- encoding = hasher.encode(data)                  pure: no registry, no recursion
        |        bytes -> leaf:  digest = BLAKE2b(0x00 + encoding)
        |        Node  -> push the children; when done:
        |                 digest = BLAKE2b(0x01 + tag + 0x00 + entries of the children)
        |                 (an entry is 0xFF + digest, or size + leaf for a leaf of at most 64 bytes)
        v
the 64-byte digest of the root  ->  hex()[:length]
```

| Decision                                                            | Why                                                                      |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `encode` returns a description, the engine hashes                   | cycles, depth, sharing, `length` and the placeholder cannot be forgotten |
| A leaf of at most 64 bytes is inlined in the preimage of its parent | a flat container makes one BLAKE2b call instead of one per item          |
| One 64-byte digest, cut at the end by `length`                      | a hasher never sees `length`; a short hash is the start of a longer one  |
| The format version is the BLAKE2b `person` parameter                | two formats can never be confused; a change is one constant              |
| A shared container is hashed once per call (a memo by `id`)         | a hash is pure, so the memo is invisible; a DAG is linear                |
| A cycle raises `ValueError`                                         | the digest of a container in a cycle would depend on where it is reached |
| Secrets are types with a hasher (`SecretHasher`)                    | the policy is the one of the registry, wherever the secret is            |
| `__coola_hash__` returns the content of a custom object, not a hash | the object cannot bypass the engine and does not forward any option      |
| Only the `compare=True` fields of a dataclass                       | equal dataclasses have the same hash                                     |

See the [user guide](../uguide/hashing.md#properties) for the properties of the package.

## Design Decisions

### Why Strict Type Checking?

**Rationale**: Prevents subtle bugs from implicit type coercion. In scientific computing, knowing
that `1` (int) and `1.0` (float) are treated differently can catch numerical issues.

**Trade-off**: Less convenient for some use cases, but more explicit and safe.

### Why Registry-Based Dispatch?

**Rationale**: Allows extensibility without modifying core code. Users can add support for their own
types.

**Trade-off**: Slightly more complex than if/else chains, but much more maintainable.

The registries are `TypeRegistry` objects (`coola.registry`). A lookup follows the Method
Resolution Order of the type, then the fallback predicates (for the types that cannot be a key, such
as every dataclass), then `object`. The handlers of the optional libraries are registered lazily
(`register_lazy`), so importing `coola` imports none of these libraries. The state of a registry is
a dictionary that is replaced on every write, so a lookup takes no lock; the registries survive
`os.fork` and `pickle`. See the [registry guide](../uguide/registry.md).

## Future Directions

Potential areas for enhancement:

1. **Parallel comparison**: For large independent comparisons
2. **Streaming comparison**: For very large objects that don't fit in memory
3. **Approximate structural matching**: For comparing objects with similar but not identical
   structure
4. **Diff generation**: Not just boolean result, but detailed diff
5. **Performance optimizations**: Cython/Numba for hot paths

## Package Layering

Packages fall into two groups:

- **Core**: `registry` (type-based dispatch), `utils` (generic helpers), `display` and
  `validation`. These must not import from any feature package.
- **Features**: `equality`, `hashing`, `summary`, `recursive`, `iterator`, `random`, `nested`,
  `reducer`, `io`, `factory`, `identifier` and `testing`. Only `equality`, `hashing`, `summary`,
  `recursive` and `iterator` are built on `registry`; `reducer` and `random` are standalone
  helpers. `identifier` depends on `hashing` (the stable identifiers are built on `hash_object`), and
  `hashing` must not import `identifier`.

There is no exception: `coola.registry` imports no feature package. `display` and `utils` currently import each other, so they are kept in the same
group.

The contract is enforced with [import-linter](https://import-linter.readthedocs.io/)
(configured in `pyproject.toml`). Run it with:

```shell
lint-imports
```

## References

- [PEP 8](https://www.python.org/dev/peps/pep-0008/): Python style guide
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [Design Patterns](https://refactoring.guru/design-patterns): Gang of Four patterns

## Contributing

To contribute to `coola`'s architecture:

1. Understand the existing patterns
2. Follow the established conventions
3. Document design decisions
4. Write tests for new components
5. Update this document for significant changes

See the [contributing guide](https://github.com/durandtibo/coola/blob/main/CONTRIBUTING.md)
for more details.
