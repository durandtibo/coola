# Testing Guide for coola

This guide documents the testing conventions and best practices used in the coola project. All contributors should follow these guidelines when writing tests.

## Table of Contents

- [Test Organization](#test-organization)
- [Test File Naming](#test-file-naming)
- [Test Function Naming](#test-function-naming)
- [Test Structure](#test-structure)
- [Imports and Dependencies](#imports-and-dependencies)
- [Fixtures](#fixtures)
- [Assertions](#assertions)
- [Parametrization](#parametrization)
- [Markers](#markers)
- [Docstrings](#docstrings)
- [Common Patterns](#common-patterns)
- [Edge Cases to Test](#edge-cases-to-test)

## Test Organization

### Directory Structure

Tests are organized into two main categories:

```
tests/
├── unit/           # Unit tests for individual components
│   ├── equality/
│   ├── iterator/
│   ├── nested/
│   ├── reducer/
│   └── utils/
└── integration/    # Integration tests for module interactions
    ├── equality/
    ├── iterator/
    └── registry/
```

- **Unit tests**: Test individual functions, classes, or methods in isolation
- **Integration tests**: Test interactions between multiple modules or components

### File Organization

Tests should be grouped logically within files using section headers:

```python
##################################
#     Tests for ClassName        #
##################################


def test_class_name_method_name_scenario() -> None: ...
```

## Test File Naming

All test files must follow the pytest convention:

- **Pattern**: `test_*.py`
- **Location**: Mirror the source code structure

Examples:
- Source: `src/coola/utils/format.py` → Test: `tests/unit/utils/test_format.py`
- Source: `src/coola/equality/handler/scalar.py` → Test: `tests/unit/equality/handler/test_scalar.py`

## Test Function Naming

Test functions should clearly describe what is being tested and under what conditions:

### Pattern

```
test_<class_or_function>_<specific_behavior>_<condition>
```

### Examples

```python
# Testing return values
def test_numpy_reducer_max_empty() -> None:
    """Test max() with empty sequence."""


# Testing initialization with specific parameters
def test_ndarray_summarizer_init_show_data_true() -> None:
    """Test NDArraySummarizer initialization with show_data=True."""


# Testing equality conditions
def test_scalar_equal_handler_equal_true() -> None:
    """Test equal() returns True for identical handlers."""


# Testing error conditions
def test_quantile_invalid_quantiles() -> None:
    """Test quantile() raises ValueError for invalid quantiles."""
```

### Naming Guidelines

- Use descriptive names that make the test purpose immediately clear
- Avoid abbreviations
- Use snake_case
- Include the expected outcome in the name (e.g., `_true`, `_false`, `_empty`, `_error`)

## Test Structure

### Standard Test Template

```python
from __future__ import annotations

import pytest

from coola.module import ClassName


@pytest.fixture
def config() -> Config:
    """Fixture for test configuration."""
    return Config()


##################################
#     Tests for ClassName        #
##################################


def test_class_name_method_default() -> None:
    """Test ClassName.method() with default parameters."""
    obj = ClassName()
    result = obj.method()
    assert result == expected_value


@pytest.mark.parametrize("param", [value1, value2, value3])
def test_class_name_method_parametrized(param: type) -> None:
    """Test ClassName.method() with various parameter values."""
    obj = ClassName()
    result = obj.method(param)
    assert result is not None
```

### Section Headers

Use hash-bounded section headers to separate test groups:

```python
##################################
#     Tests for ClassName        #
##################################
```

- Pattern: `# [#'s][test name][#'s]`
- Visually separates tests for different classes/components
- Maintains consistent spacing (4 spaces on each side)

## Imports and Dependencies

### Required Imports

Always include the future annotations import:

```python
from __future__ import annotations
```

### Type Checking Imports

Use TYPE_CHECKING for imports only needed for type hints:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
```

### Optional Dependencies

Handle optional dependencies using mocks:

```python
from unittest.mock import Mock

from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()  # pragma: no cover
```

## Fixtures

### Defining Fixtures

Use pytest fixtures for reusable test setup:

```python
@pytest.fixture
def config() -> EqualityConfig:
    """Return a default EqualityConfig for testing."""
    return EqualityConfig()


@pytest.fixture
def registry() -> SummarizerRegistry:
    """Return a default SummarizerRegistry for testing."""
    return SummarizerRegistry()
```

### Auto-use Fixtures

Use `autouse=True` for fixtures that should run automatically:

```python
@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the default registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
```

### Shared Test Data

Create helper files for shared test data:

```python
# tests/unit/equality/utils.py
from dataclasses import dataclass


@dataclass
class ExamplePair:
    """A pair of objects for equality testing."""

    actual: object
    expected: object


SCALAR_EQUAL = [
    ExamplePair(actual=1, expected=1),
    ExamplePair(actual=1.0, expected=1.0),
    ExamplePair(actual="abc", expected="abc"),
]
```

## Assertions

### Standard Assertions

Use simple pytest assertions:

```python
# Equality
assert result == expected

# Identity
assert result is None

# Type checking
assert isinstance(result, int)

# Boolean conditions
assert condition
assert not condition

# String representation
assert repr(obj) == "ClassName(param=value)"
```

### Exception Assertions

Use `pytest.raises()` for exception testing:

```python
def test_function_raises_error() -> None:
    """Test function raises ValueError for invalid input."""
    with pytest.raises(ValueError, match=r"invalid input"):
        function_that_raises(invalid_input)
```

### Logging Assertions

Use the `caplog` fixture for log output verification:

```python
def test_function_logs_warning(caplog: LogCaptureFixture) -> None:
    """Test function logs a warning message."""
    with caplog.at_level(logging.WARNING):
        function_that_logs()
        assert "expected warning" in caplog.messages[0]
```

## Parametrization

### Basic Parametrization

Use `@pytest.mark.parametrize` to test multiple inputs:

```python
@pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
def test_function_with_various_values(value: int) -> None:
    """Test function with various integer values."""
    result = function(value)
    assert result > 0
```

### Parametrization with IDs

Use `pytest.param()` with `id` for readable test names:

```python
@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            NanEqualHandler(),
            NanEqualHandler(),
            id="without next handler",
        ),
        pytest.param(
            NanEqualHandler(FalseHandler()),
            NanEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_handlers_are_equal(
    handler1: NanEqualHandler,
    handler2: NanEqualHandler,
) -> None:
    """Test equality between handlers."""
    assert handler1.equal(handler2)
```

### Multi-Parameter Parametrization

Stack multiple `@pytest.mark.parametrize` decorators:

```python
@pytest.mark.parametrize("function", [func1, func2, func3])
@pytest.mark.parametrize("example", [example1, example2])
@pytest.mark.parametrize("show_difference", [True, False])
def test_function_combinations(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
) -> None:
    """Test all combinations of parameters."""
    assert function(example.actual, example.expected, show_difference=show_difference)
```

### Shared Parameter Sets

Define parameter sets in helper files for reuse:

```python
# In helpers.py
DEFAULT_SAMPLES = [
    pytest.param("abc", [], id="str"),
    pytest.param(42, [], id="int"),
    pytest.param(None, [None], id="none"),
]


# In test file
@pytest.mark.parametrize(("data", "expected"), DEFAULT_SAMPLES)
def test_function(data: object, expected: object) -> None:
    """Test function with shared parameter sets."""
    assert process(data) == expected
```

## Markers

### Parametrize Marker

The `@pytest.mark.parametrize` marker is used extensively (see [Parametrization](#parametrization) section).

### Optional Dependency Markers

Use custom markers for tests requiring optional dependencies:

```python
from coola.testing.fixtures import numpy_available


@numpy_available
def test_numpy_feature() -> None:
    """Test feature that requires NumPy."""
    import numpy as np

    array = np.array([1, 2, 3])
    assert len(array) == 3
```

Available markers:
- `@numpy_available` - Requires NumPy
- `@torch_available` - Requires PyTorch
- Similar markers exist for other optional dependencies

## Docstrings

### When to Use Docstrings

Add docstrings to tests when:
- The test is complex and the name alone isn't self-explanatory
- The test demonstrates important behavior
- The test is part of a critical module (e.g., equality, summary)

### Docstring Format

Use single-line docstrings for most tests:

```python
def test_ndarray_summarizer_init_default() -> None:
    """Test NDArraySummarizer initialization with default parameters."""
    summarizer = NDArraySummarizer()
    assert summarizer._show_data is False
```

### When to Omit Docstrings

Omit docstrings when the test name is self-documenting:

```python
def test_nan_equal_handler_repr() -> None:
    assert repr(NanEqualHandler()) == "NanEqualHandler()"
```

## Common Patterns

### Testing `__repr__` and `__str__`

```python
def test_class_repr() -> None:
    """Test __repr__ returns expected string representation."""
    obj = ClassName(param=value)
    assert repr(obj) == "ClassName(param=value)"


def test_class_str() -> None:
    """Test __str__ returns expected string."""
    obj = ClassName(param=value)
    assert str(obj) == "ClassName(param=value)"
```

### Testing `equal()` Methods

```python
def test_class_equal_same_instance() -> None:
    """Test equal() returns True for same instance."""
    obj = ClassName()
    assert obj.equal(obj)


def test_class_equal_same_config() -> None:
    """Test equal() returns True for objects with same configuration."""
    obj1 = ClassName(param=value)
    obj2 = ClassName(param=value)
    assert obj1.equal(obj2)


def test_class_equal_different_param() -> None:
    """Test equal() returns False for objects with different parameters."""
    obj1 = ClassName(param=value1)
    obj2 = ClassName(param=value2)
    assert not obj1.equal(obj2)


def test_class_equal_different_type() -> None:
    """Test equal() returns False when comparing with different type."""
    assert not ClassName().equal(42)
```

### Testing Inheritance

```python
def test_class_equal_false_different_type_child() -> None:
    """Test equal() returns False for child classes."""

    class Child(ClassName):
        pass

    assert not ClassName().equal(Child())
```

## Edge Cases to Test

Always consider testing these edge cases:

### Input Validation

- Empty sequences: `[]`, `()`, `{}`
- Single-element sequences: `[1]`
- `None` values
- Invalid types
- Boundary values (min/max, zero, negative)

### Numeric Edge Cases

```python
def test_function_empty_sequence() -> None:
    """Test function with empty sequence."""
    result = function([])
    assert result is expected


def test_function_single_value() -> None:
    """Test function with single-value sequence."""
    result = function([5])
    assert result == 5


def test_function_negative_values() -> None:
    """Test function with negative values."""
    result = function([-5, -3, -1])
    assert result == expected


def test_function_zero() -> None:
    """Test function with zero value."""
    result = function(0)
    assert result == expected


def test_function_none() -> None:
    """Test function with None input."""
    with pytest.raises(TypeError):
        function(None)
```

### String Edge Cases

```python
def test_function_empty_string() -> None:
    """Test function with empty string."""
    result = function("")
    assert result == expected


def test_function_whitespace() -> None:
    """Test function with whitespace-only string."""
    result = function("   ")
    assert result == expected


def test_function_special_characters() -> None:
    """Test function with special characters."""
    result = function("hello\nworld\ttab")
    assert result == expected
```

### Collection Edge Cases

```python
def test_function_empty_dict() -> None:
    """Test function with empty dictionary."""
    result = function({})
    assert result == expected


def test_function_nested_empty() -> None:
    """Test function with nested empty structures."""
    result = function([[]])
    assert result == expected


def test_function_mixed_types() -> None:
    """Test function with mixed types."""
    result = function([1, "a", None, 2.5])
    assert result == expected
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Unit Tests Only

```bash
pytest tests/unit/
```

### Run Integration Tests Only

```bash
pytest tests/integration/
```

### Run Tests for Specific Module

```bash
pytest tests/unit/equality/
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=coola --cov-report=html
```

## Best Practices

1. **Keep tests focused**: Each test should verify one specific behavior
2. **Use descriptive names**: Test names should clearly indicate what is being tested
3. **Avoid test interdependencies**: Tests should be independent and runnable in any order
4. **Test edge cases**: Always consider empty inputs, None values, and boundary conditions
5. **Use fixtures for setup**: Avoid repeating setup code across tests
6. **Use parametrization**: Test multiple inputs with a single test function
7. **Test error conditions**: Verify that functions raise appropriate exceptions
8. **Keep tests maintainable**: Use helper functions and shared test data
9. **Follow existing patterns**: Maintain consistency with the existing test suite
10. **Document complex tests**: Add docstrings when the test logic isn't immediately obvious
