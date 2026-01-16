# Migration Guide: v0.11 to v1.0

This guide explains how to migrate your code from `coola` version `0.11.x` to version `1.0.x`.
Version 1.0 includes significant API reorganization to improve maintainability and extensibility.

## Overview of Changes

The main changes in version 1.0 are:

1. **Module reorganization**: Better structured package hierarchy
2. **Import path changes**: Top-level imports moved to subpackages
3. **Naming changes**: More consistent naming conventions (e.g., `comparators` → `tester`)
4. **Removed features**: Some modules were removed or merged (e.g., `formatters`)
5. **New architecture**: Introduction of handlers pattern for equality checking

## Quick Migration Checklist

- [ ] Update imports for equality functions (`coola` → `coola.equality`)
- [ ] Update imports for summarization (`coola.summarizers` → `coola.summary`)
- [ ] Replace uses of formatters with summary functions
- [ ] Update custom comparators to use new tester/handler pattern
- [ ] Update module names (`reduction` → `reducer`)

## Detailed Changes

### 1. Top-Level Imports Removed

The main equality functions are no longer available at the top level.

**Before (v0.11):**

```python
from coola import objects_are_equal, objects_are_allclose
```

**After (v1.0):**

```python
from coola.equality import objects_are_equal, objects_are_allclose
```

**Migration:** Add `.equality` to your import statements.

### 2. Module Reorganization

Several modules have been renamed or reorganized:

| Feature | v0.11 | v1.0 | Status |
|---------|-------|------|--------|
| Equality functions module | `coola.comparison` (internal) | `coola.equality` | Renamed |
| Comparators | `coola.equality.comparators` | `coola.equality.tester` | Renamed |
| Formatters | `coola.formatters` | ❌ Removed | Use `coola.summary` instead |
| Summarizers | `coola.summarizers` | `coola.summary` | Renamed |
| Reduction | `coola.reduction` | `coola.reducer` | Renamed |

**Note:** In v0.11, equality functions were available both at the top level (`from coola import`) and in `coola.comparison` module. In v1.0, they are only in `coola.equality`.

### 3. Equality Comparison System

The equality comparison system has been redesigned with a cleaner architecture.

#### Basic Usage (No Breaking Changes)

If you only use the basic equality functions, the migration is simple:

**Before (v0.11):**

```python
from coola import objects_are_equal, objects_are_allclose

objects_are_equal(actual, expected)
objects_are_allclose(actual, expected, rtol=1e-5, atol=1e-8)
```

**After (v1.0):**

```python
from coola.equality import objects_are_equal, objects_are_allclose

objects_are_equal(actual, expected)
objects_are_allclose(actual, expected, rtol=1e-5, atol=1e-8)
```

The function signatures and behavior remain the same.

#### Custom Equality Testers

The comparator system has been replaced with a tester + handler architecture.

**Before (v0.11):**

```python
from coola.equality.comparators import BaseEqualityComparator
from coola.equality.testers import EqualityTester

# Custom comparator
class MyComparator(BaseEqualityComparator):
    # implementation
    pass

# Register and use
tester = EqualityTester()
# ...
```

**After (v1.0):**

```python
from coola.equality.tester import BaseEqualityTester
from coola.equality.handler import BaseEqualityHandler

# Custom tester
class MyTester(BaseEqualityTester):
    # implementation
    pass

# Or use handlers for common patterns
class MyHandler(BaseEqualityHandler):
    # implementation
    pass
```

**Migration:**

1. Rename `Comparator` classes to `Tester` classes
2. Update import paths: `coola.equality.comparators` → `coola.equality.tester`
3. Consider using handlers for reusable comparison logic

#### Type-Specific Classes

**Before (v0.11):**

```python
from coola.equality.comparators import (
    TorchTensorComparator,
    NumpyArrayComparator,
    MappingEqualityComparator,
    SequenceEqualityComparator,
)
```

**After (v1.0):**

```python
from coola.equality.tester import (
    TorchTensorEqualityTester,
    NumpyArrayEqualityTester,
    MappingEqualityTester,
    SequenceEqualityTester,
)
```

**Migration:** Replace `Comparator` with `EqualityTester` in class names.

### 4. Formatters Removed

The `coola.formatters` module has been removed. Use `coola.summary` for text representation instead.

**Before (v0.11):**

```python
from coola.formatters import Formatter, DefaultFormatter

formatter = DefaultFormatter()
formatted_output = formatter.format(obj)
```

**After (v1.0):**

```python
from coola.summary import summarize

summary_output = summarize(obj)
```

**Migration:**

- Replace formatter calls with `summarize()` function
- The `summarize()` function provides similar text representation functionality
- If you need custom formatting logic, implement a custom `BaseSummarizer`

### 5. Summarization

The summarization module has been renamed from `summarizers` to `summary`.

**Before (v0.11):**

```python
from coola import summary

summary_text = summary(value, max_depth=2)
```

**After (v1.0):**

```python
from coola.summary import summarize

summary_text = summarize(value, max_depth=2)
```

**Changes:**

- Module renamed: `coola.summarizers` → `coola.summary`
- Function renamed: `summary()` → `summarize()`
- Top-level import removed: use `from coola.summary import summarize`

### 6. Reduction

The reduction module has been renamed from `reduction` to `reducer`.

**Before (v0.11):**

```python
from coola.reduction import Reduction

Reduction.reducer = some_reducer
```

**After (v1.0):**

```python
from coola.reducer import auto_reducer, BaseReducer, NativeReducer

reducer = auto_reducer()
# Or use specific reducers:
# NativeReducer(), NumpyReducer(), TorchReducer()
```

**Migration:**

- Update module name: `coola.reduction` → `coola.reducer`
- The `Reduction` class pattern has been replaced with direct reducer instances
- Use `auto_reducer()` for automatic backend selection

### 7. New Features in v1.0

Several new modules have been added in v1.0:

- **`coola.equality.handler`**: Chain of responsibility pattern for equality checks
- **`coola.nested`**: Utilities for working with nested data structures
- **`coola.iterator`**: DFS/BFS traversal of nested structures
- **`coola.recursive`**: Recursive processing utilities
- **`coola.registry`**: Generic registry implementations
- **`coola.random`**: Random data generation utilities

These are new additions and don't require migration from v0.11.

## Common Migration Patterns

### Pattern 1: Simple Equality Checks

**Before:**

```python
from coola import objects_are_equal
import torch

result = objects_are_equal(
    {"a": torch.tensor([1, 2, 3])},
    {"a": torch.tensor([1, 2, 3])}
)
```

**After:**

```python
from coola.equality import objects_are_equal
import torch

result = objects_are_equal(
    {"a": torch.tensor([1, 2, 3])},
    {"a": torch.tensor([1, 2, 3])}
)
```

### Pattern 2: Tolerance-Based Comparison

**Before:**

```python
from coola import objects_are_allclose
import numpy as np

result = objects_are_allclose(
    np.array([1.0, 2.0]),
    np.array([1.0001, 2.0001]),
    atol=1e-3
)
```

**After:**

```python
from coola.equality import objects_are_allclose
import numpy as np

result = objects_are_allclose(
    np.array([1.0, 2.0]),
    np.array([1.0001, 2.0001]),
    atol=1e-3
)
```

### Pattern 3: Summarization

**Before:**

```python
from coola import summary

text = summary([1, 2, 3, {"key": "value"}], max_depth=2)
print(text)
```

**After:**

```python
from coola.summary import summarize

text = summarize([1, 2, 3, {"key": "value"}], max_depth=2)
print(text)
```

### Pattern 4: Custom Comparators/Testers

**Before:**

```python
from coola.equality.comparators import BaseEqualityComparator
from coola.equality.config import EqualityConfig

class MyCustomComparator(BaseEqualityComparator):
    def __eq__(self, other: object) -> bool:
        # implementation
        pass
    
    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        # comparison logic
        pass
```

**After:**

```python
from coola.equality.tester import BaseEqualityTester
from coola.equality.config import EqualityConfig

class MyCustomTester(BaseEqualityTester):
    def __eq__(self, other: object) -> bool:
        # implementation
        pass
    
    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        # comparison logic
        pass
```

## Testing Your Migration

After migrating your code, ensure everything works correctly:

1. **Update imports**: Run your code to check for import errors
2. **Run tests**: Ensure all existing tests pass
3. **Check functionality**: Verify equality checks produce the same results
4. **Review custom code**: If you have custom comparators/testers, review and test thoroughly

## Getting Help

If you encounter issues during migration:

- Check the [documentation](https://durandtibo.github.io/coola/) for detailed API reference
- Review the [examples](https://durandtibo.github.io/coola/uguide/quickstart/) for common usage patterns
- Open an issue on [GitHub](https://github.com/durandtibo/coola/issues) if you need help

## API Stability Note

⚠️ **Important**: This migration guide is for the 1.0 release series. The current version is `1.0.0a0` (alpha release).

- The API structure described in this guide reflects the 1.0 design
- The changes from v0.11 documented here are stable and will remain in the 1.0 series
- Minor refinements may occur before the final 1.0.0 stable release
- For production use, monitor the [releases page](https://github.com/durandtibo/coola/releases) and consider pinning to a specific version
