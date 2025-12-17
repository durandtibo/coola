# Architecture and Design

This document describes the internal architecture and design principles of `coola`.

## Overview

`coola` is designed around a flexible, extensible comparison framework that can handle various data types through a plugin-like architecture. The core design follows these principles:

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

Testers are responsible for orchestrating the comparison process.

#### `BaseEqualityTester`

Abstract base class defining the tester interface:

```python
class BaseEqualityTester:
    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        """Check if two objects are equal."""
        ...
```

#### `EqualityTester`

The default implementation that uses a registry of comparators:

- Maintains a registry mapping types to comparators
- Uses Method Resolution Order (MRO) to find the most specific comparator
- Delegates comparison to the appropriate comparator

**Key Features:**
- Type-based dispatch
- Support for inheritance hierarchies
- Extensible through registration

### 3. Comparators

Comparators implement comparison logic for specific types.

#### `BaseEqualityComparator`

Abstract base class for comparators:

```python
class BaseEqualityComparator:
    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        """Compare two objects of a specific type."""
        ...

    def clone(self) -> BaseEqualityComparator:
        """Create a copy of this comparator."""
        ...
```

#### Built-in Comparators

- **`DefaultEqualityComparator`**: Handles basic Python types
- **`MappingEqualityComparator`**: Handles dict and mapping types
- **`SequenceEqualityComparator`**: Handles list, tuple, and sequences
- **`TorchTensorComparator`**: Handles PyTorch tensors
- **`NumpyArrayComparator`**: Handles NumPy arrays
- **`PandasDataFrameComparator`**: Handles pandas DataFrames
- And more for other supported types...

### 4. Configuration

#### `EqualityConfig`

Configuration object that carries comparison settings through the comparison tree:

```python
@dataclasses.dataclass
class EqualityConfig:
    tester: BaseEqualityTester
    show_difference: bool = False
    # Additional settings...
```

This allows behavior to be customized without changing comparator signatures.

### 5. Handlers

Handlers provide reusable comparison logic for common checks:

- **`DTypeHandler`**: Compares data types
- **`ShapeHandler`**: Compares array shapes
- **`DeviceHandler`**: Compares PyTorch device placement
- **`NativeEqualHandler`**: Performs native equality checks
- And more...

Handlers promote code reuse and consistency across comparators.

## Data Flow

Here's how a comparison flows through the system:

```
User calls objects_are_equal(obj1, obj2)
    ↓
Creates EqualityConfig with settings
    ↓
Calls tester.equal(obj1, obj2, config)
    ↓
Tester looks up comparator based on obj1's type
    ↓
Calls comparator.equal(obj1, obj2, config)
    ↓
Comparator performs type-specific checks
    ↓
May recursively call tester.equal() for nested objects
    ↓
Returns boolean result
```

## Design Patterns

### 1. Strategy Pattern

Comparators implement different comparison strategies for different types, allowing the algorithm to vary independently from the clients that use it.

### 2. Chain of Responsibility

The MRO-based comparator lookup implements a chain of responsibility, trying more specific comparators before falling back to general ones.

### 3. Template Method

Many comparators follow a template:
1. Check types match
2. Check metadata (shape, dtype, etc.)
3. Check values
4. Optionally show differences

### 4. Registry Pattern

The comparator registry allows runtime type-to-comparator mapping, enabling extensibility.

### 5. Visitor Pattern

The recursive nature of comparison through nested structures follows a visitor-like pattern.

## Extension Points

### Adding Support for New Types

To add support for a custom type:

1. **Implement a Comparator:**
   ```python
   class MyTypeComparator(BaseEqualityComparator):
       def equal(self, actual: MyType, expected: Any, config: EqualityConfig) -> bool:
           # Type check
           if type(actual) is not type(expected):
               return False

           # Custom comparison logic
           return actual.compare_to(expected)

       def clone(self):
           return MyTypeComparator()
   ```

2. **Register the Comparator:**
   ```python
   tester = EqualityTester.local_copy()
   tester.add_comparator(MyType, MyTypeComparator())
   ```

3. **Use with Custom Tester:**
   ```python
   objects_are_equal(obj1, obj2, tester=tester)
   ```

## Type System

### Strict Type Checking

`coola` enforces strict type checking:
- `1` (int) ≠ `1.0` (float) ≠ `True` (bool)
- `list` ≠ `tuple`
- `dict` ≠ `OrderedDict`

This prevents subtle bugs from type coercion.

### Type Hierarchy Support

Through MRO-based lookup, `coola` supports inheritance:
- A comparator for `Sequence` applies to `list`, `tuple`, etc.
- More specific comparators override general ones
- Custom subclasses inherit parent comparators

## Performance Considerations

### Early Exit

Comparators check fast properties first:
1. Type check (very fast)
2. Metadata checks (fast: shape, dtype, device)
3. Value comparison (potentially slow)

### Lazy Evaluation

Comparisons short-circuit on first difference when possible.

### Caching

The tester caches comparator lookups by type for performance.

### Recursive Depth

For deeply nested structures, comparison is recursive. Very deep nesting may hit recursion limits (typically ~1000 levels in Python).

## Error Handling

### Graceful Degradation

When a specific comparator is not available, `coola` falls back to:
1. More general comparator (via MRO)
2. Default comparator (for `object`)
3. Native equality check as last resort

### Informative Messages

When `show_difference=True`, comparators log:
- What objects differ
- Where in the structure the difference is
- The actual values that differ

## Testing Strategy

The `coola` codebase uses:

1. **Unit tests**: Test individual comparators in isolation
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
├── comparison.py          # Main public API
├── equality/
│   ├── comparators/       # Type-specific comparators
│   │   ├── base.py
│   │   ├── default.py
│   │   ├── collection.py  # Mapping, Sequence
│   │   ├── torch_.py
│   │   ├── numpy_.py
│   │   └── ...
│   ├── testers/          # Comparison orchestration
│   │   ├── base.py
│   │   └── default.py
│   ├── handlers/         # Reusable comparison logic
│   └── config.py         # Configuration
├── allclose/             # Tolerance-based comparison
└── utils/                # Utility functions
```

## Design Decisions

### Why Strict Type Checking?

**Rationale**: Prevents subtle bugs from implicit type coercion. In scientific computing, knowing that `1` (int) and `1.0` (float) are treated differently can catch numerical issues.

**Trade-off**: Less convenient for some use cases, but more explicit and safe.

### Why Registry-Based Dispatch?

**Rationale**: Allows extensibility without modifying core code. Users can add support for their own types.

**Trade-off**: Slightly more complex than if/else chains, but much more maintainable.

### Why Separate Testers and Comparators?

**Rationale**: Separation of concerns. Testers handle dispatch and orchestration, comparators handle type-specific logic.

**Trade-off**: More classes/files, but better modularity.

### Why Handlers?

**Rationale**: Code reuse. Many comparators need similar checks (dtype, shape, etc.).

**Trade-off**: One more abstraction layer, but reduces duplication.

## Future Directions

Potential areas for enhancement:

1. **Parallel comparison**: For large independent comparisons
2. **Streaming comparison**: For very large objects that don't fit in memory
3. **Approximate structural matching**: For comparing objects with similar but not identical structure
4. **Diff generation**: Not just boolean result, but detailed diff
5. **Performance optimizations**: Cython/Numba for hot paths

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

See the [contributing guide](https://github.com/durandtibo/coola/blob/main/.github/CONTRIBUTING.md) for more details.
