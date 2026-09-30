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
    def equal(self, other: object) -> bool:
        """Indicate if ``other`` is a tester of the same type."""

    @abstractmethod
    def objects_are_equal(self, actual: T, expected: object, config: EqualityConfig) -> bool:
        """Indicate if two objects are equal."""
```

#### Built-in Testers

- **`DefaultEqualityTester`**: Fallback for any type (identity, type check, then `==`)
- **`MappingEqualityTester`** and **`SequenceEqualityTester`**: Recurse into mappings and sequences
- **`ScalarEqualityTester`**, **`EqualEqualityTester`**, **`EqualNanEqualityTester`**,
  **`TolerantEqualEqualityTester`**: Scalars and objects with an `equal` method
- Array and dataframe testers for NumPy, PyTorch, pandas, polars, xarray, JAX and PyArrow (for
  example `NumpyArrayEqualityTester`, `TorchTensorEqualityTester`,
  `PandasDataFrameEqualityTester`)
- **`HandlerEqualityTester`**: Wraps a chain of handlers (see below)

### 3. Registry

#### `EqualityTesterRegistry`

The registry maps types to testers. It is built on `BaseTypeDispatchRegistry`
(`coola.registry`):

- `find_equality_tester(data_type)` returns the tester for the most specific registered type,
  walking the Method Resolution Order (MRO); `object` is registered with `DefaultEqualityTester`,
  so a tester is always found
- `objects_are_equal(actual, expected, config)` finds the tester for `type(actual)` and delegates
  to it
- Lookup results are cached per type, and the cache is cleared whenever the registry changes

`get_default_registry()` returns the global registry. Testers for optional libraries are only
registered when the library is installed. `register_equality_testers(mapping, exist_ok=False)`
adds testers to it.

### 4. Configuration

#### `EqualityConfig`

A dataclass that carries the comparison settings through the comparison tree:

```python
@dataclass
class EqualityConfig:
    registry: EqualityTesterRegistry  # defaults to the default registry
    equal_nan: bool = False
    atol: float = 0.0
    rtol: float = 0.0
    show_difference: bool = False
    max_depth: int = 1000
```

This allows behavior to be customized without changing tester signatures. A config tracks the
current recursion depth, so it is not thread-safe: create one config per comparison.

### 5. Handlers

Handlers (`coola.equality.handler`) are small reusable checks that are chained together
(chain of responsibility). Each handler either returns a result or passes the comparison to the
next handler in the chain. Examples:

- **`SameObjectHandler`**, **`SameTypeHandler`**, **`SameLengthHandler`**: Generic checks
- **`SameDTypeHandler`**, **`SameShapeHandler`**: Metadata checks
- **`TorchTensorSameDeviceHandler`**: Compares PyTorch device placement
- **`ObjectEqualHandler`**, **`NanEqualHandler`**, **`TolerantEqualHandler`**: Value checks
- **`MappingSameKeysHandler`**, **`MappingSameValuesHandler`**, **`SequenceSameValuesHandler`**:
  Recursive checks for containers
- Library-specific handlers such as `NumpyArrayEqualHandler` or `PandasDataFrameEqualHandler`

`create_chain(*handlers)` links handlers and returns the first one. A `HandlerEqualityTester`
turns a chain into a tester. Handlers promote code reuse and consistency across testers.

### 6. Comparison Results

`compare` returns a `ComparisonResult`, and `assert_objects_equal` / `assert_objects_allclose`
raise an `AssertionError` with a description of the difference.

## Data Flow

Here's how a comparison flows through the system:

```
User calls objects_are_equal(obj1, obj2)
    ↓
Creates EqualityConfig with settings (and the registry)
    ↓
Calls registry.objects_are_equal(obj1, obj2, config)
    ↓
Registry looks up the tester based on type(obj1) (MRO lookup)
    ↓
Calls tester.objects_are_equal(obj1, obj2, config)
    ↓
Tester runs its handler chain (type check, metadata, values)
    ↓
May recursively call registry.objects_are_equal() for nested objects
    ↓
Returns boolean result
```

## Design Patterns

### 1. Strategy Pattern

Testers implement different comparison strategies for different types, allowing the algorithm to
vary independently from the clients that use it.

### 2. Chain of Responsibility

Handlers are chained: each one either decides the result or passes the comparison to the next
handler. The MRO-based tester lookup also tries more specific testers before falling back to
general ones.

### 3. Template Method

Many testers follow a template:

1. Check types match
2. Check metadata (shape, dtype, etc.)
3. Check values
4. Optionally show differences

### 4. Registry Pattern

The tester registry allows runtime type-to-tester mapping, enabling extensibility.

### 5. Visitor Pattern

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

## Type System

### Strict Type Checking

`coola` enforces strict type checking:

- `1` (int) ≠ `1.0` (float) ≠ `True` (bool)
- `list` ≠ `tuple`
- `dict` ≠ `OrderedDict`

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

### Recursive Depth

For deeply nested structures, comparison is recursive. `EqualityConfig.max_depth` (default 1000)
bounds the nesting depth. Each level uses several interpreter frames, so Python's own recursion
limit (`sys.setrecursionlimit`) may be reached first, in which case a `RecursionError` with an
actionable message is raised.

## Error Handling

### Graceful Degradation

When a specific tester is not available, `coola` falls back to:

1. More general tester (via MRO)
2. `DefaultEqualityTester` (registered for `object`), which uses `==`

### Informative Messages

When `show_difference=True`, testers log:

- What objects differ
- Where in the structure the difference is
- The actual values that differ

## Testing Strategy

The `coola` codebase uses:

1. **Unit tests**: Test individual testers and handlers in isolation
2. **Integration tests**: Test complete comparison workflows
3. **Property-based tests**: Test invariants (e.g., reflexivity)
4. **Cross-library tests**: Test integration with PyTorch, NumPy, etc.

## Dependencies

### Core Dependencies

- Python 3.10+: Core language features

### Optional Dependencies

- **torch**: PyTorch tensor support
- **numpy**: NumPy array support
- **pandas**: DataFrame support
- **polars**: Polars DataFrame support
- **xarray**: xarray support
- **jax**: JAX array support
- **pyarrow**: PyArrow table support

Each optional dependency is only imported when used (lazy loading).

## Module Organization

```
coola/
├── equality/             # Equality and tolerance comparison
│   ├── interface.py      # objects_are_equal, objects_are_allclose
│   ├── result.py         # compare, assert_objects_equal, ...
│   ├── config.py         # EqualityConfig
│   ├── tester/           # Type-specific testers and the registry
│   └── handler/          # Reusable comparison logic
├── registry/             # Generic registries and type dispatch
├── summary/, hashing/, recursive/, iterator/, nested/, random/, reducer/
├── io/, factory/, identifier/, display/, validation/, testing/
└── utils/                # Utility functions
```

## Design Decisions

### Why Strict Type Checking?

**Rationale**: Prevents subtle bugs from implicit type coercion. In scientific computing, knowing
that `1` (int) and `1.0` (float) are treated differently can catch numerical issues.

**Trade-off**: Less convenient for some use cases, but more explicit and safe.

### Why Registry-Based Dispatch?

**Rationale**: Allows extensibility without modifying core code. Users can add support for their own
types.

**Trade-off**: Slightly more complex than if/else chains, but much more maintainable.

### Why Separate Testers and Handlers?

**Rationale**: Separation of concerns. Testers are the per-type entry points found by the registry,
handlers are small reusable checks that testers chain together.

**Trade-off**: More classes/files, but better modularity.

### Why Handlers?

**Rationale**: Code reuse. Many testers need similar checks (dtype, shape, etc.).

**Trade-off**: One more abstraction layer, but reduces duplication.

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
  `recursive` and `iterator` are built on `registry`; `identifier`, `reducer` and `random` are
  standalone helpers.

The only exception is a deferred (function-level) import in `coola.registry.base`, which is listed in
`ignore_imports`. `display` and `utils` currently import each other, so they are kept in the same
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
