# Frequently Asked Questions (FAQ)

## General Questions

### What is `coola`?

`coola` is a Python library that provides simple functions to check if two complex/nested objects
are equal or not. It was initially designed to work with PyTorch `Tensor`s and NumPy `ndarray`s, but
it can be extended to support other data structures.

### Why use `coola` instead of `==` operator?

The native Python equality operator `==` cannot handle complex nested structures containing objects
like PyTorch tensors or NumPy arrays. `coola` provides a unified interface to compare these complex
structures in a single line of code.

### Is `coola` production-ready?

`coola` is actively developed and maintained. However, it is still in
development (version 0.x), which means the API may change between releases. See
the [API stability](index.md#api-stability) section for more information.

## Installation Questions

### How do I install `coola` with all dependencies?

You can install `coola` with all optional dependencies using:

```shell
pip install 'coola[all]'
```

### Can I install only specific dependencies?

Yes! You can install only the dependencies you need. For example, to install only NumPy support:

```shell
pip install coola numpy
# or
pip install 'coola[numpy]'
```

### What Python versions are supported?

`coola` supports Python 3.10 and above. See
the [compatibility table](https://github.com/durandtibo/coola#installation) for detailed version
support.

## Usage Questions

### How do I compare PyTorch tensors?

```python
import torch
from coola import objects_are_equal

tensor1 = torch.ones(2, 3)
tensor2 = torch.ones(2, 3)
result = objects_are_equal(tensor1, tensor2)  # Returns True
```

### How do I compare NumPy arrays?

```python
import numpy as np
from coola import objects_are_equal

array1 = np.ones((2, 3))
array2 = np.ones((2, 3))
result = objects_are_equal(array1, array2)  # Returns True
```

### How do I compare nested dictionaries with mixed types?

```python
import torch
import numpy as np
from coola import objects_are_equal

data1 = {
    "tensor": torch.ones(2, 3),
    "array": np.zeros((2, 3)),
    "list": [1, 2, 3],
    "nested": {"key": "value"},
}
data2 = {
    "tensor": torch.ones(2, 3),
    "array": np.zeros((2, 3)),
    "list": [1, 2, 3],
    "nested": {"key": "value"},
}
result = objects_are_equal(data1, data2)  # Returns True
```

### What's the difference between `objects_are_equal` and `objects_are_allclose`?

- `objects_are_equal`: Requires exact equality of values
- `objects_are_allclose`: Allows small numerical differences within a tolerance (useful for
  floating-point comparisons)

```python
import torch
from coola import objects_are_equal, objects_are_allclose

tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([1.0, 2.0, 3.00001])

objects_are_equal(tensor1, tensor2)  # Returns False
objects_are_allclose(tensor1, tensor2)  # Returns True (within default tolerance)
```

### How do I see which elements are different?

Use the `show_difference=True` parameter:

```python
from coola import objects_are_equal

data1 = {"a": 1, "b": 2, "c": 3}
data2 = {"a": 1, "b": 99, "c": 3}

result = objects_are_equal(data1, data2, show_difference=True)
# This will log the first difference found
```

### Why does `coola` say 1 and 1.0 are different?

Unlike Python's native `==` operator, `coola` uses strict type checking. This means:

- `1` (int) is different from `1.0` (float)
- `1` (int) is different from `True` (bool)
- `dict` is different from `collections.OrderedDict`

This design choice prevents subtle bugs from type mismatches.

### How do I handle NaN values?

By default, NaN values are not considered equal (following NumPy conventions). To treat NaN values
as equal, use the `equal_nan=True` parameter:

```python
import numpy as np
from coola import objects_are_allclose

array1 = np.array([1.0, 2.0, float("nan")])
array2 = np.array([1.0, 2.0, float("nan")])

objects_are_allclose(array1, array2)  # Returns False
objects_are_allclose(array1, array2, equal_nan=True)  # Returns True
```

## Customization Questions

### Can I add support for custom types?

Yes! `coola` is designed to be extensible. You can implement custom comparators for your own types.
See the [customization guide](customization.md) for details.

### How do I customize comparison behavior?

You can implement a custom `BaseEqualityComparator` or extend the default `EqualityTester`. See
the [customization documentation](customization.md) for examples.

## Performance Questions

### How fast is `coola`?

Performance depends on the complexity and size of the objects being compared. For small to
medium-sized objects, the overhead is minimal. For very large objects, consider:

- Comparing only necessary fields
- Using `objects_are_allclose` with appropriate tolerances
- Implementing custom comparators for performance-critical paths

### Can I compare very large objects?

Yes, but be aware that comparing very large or deeply nested structures may consume significant
memory and CPU. Consider:

- Breaking down comparisons into smaller chunks
- Using sampling for very large datasets
- Implementing early-exit strategies in custom comparators

### Does `coola` support parallel comparison?

Currently, `coola` performs comparisons sequentially. Parallel comparison is not supported out of
the box, but you can implement custom comparators with parallel logic if needed.

## Troubleshooting Questions

### Why does comparison fail for tensors on different devices?

`coola` considers tensors on different devices (CPU vs GPU) as different, even if their values are
identical. This is by design to catch device mismatches. To compare values regardless of device,
move tensors to the same device first:

```python
import torch
from coola import objects_are_equal

tensor1 = torch.ones(2, 3)  # CPU
tensor2 = torch.ones(2, 3, device="cuda")  # GPU

# Move to same device before comparison
objects_are_equal(tensor1, tensor2.cpu())  # Returns True
```

## Integration Questions

### Can I use `coola` with pytest?

Yes! `coola` works great with pytest. You can use it in assertions:

```python
import torch
from coola import objects_are_equal


def test_my_function():
    result = my_function()
    expected = {"tensor": torch.ones(2, 3), "value": 42}
    assert objects_are_equal(result, expected, show_difference=True)
```

### Does `coola` work with dataclasses?

Yes! Dataclasses are compared like regular objects:

```python
from dataclasses import dataclass
from coola import objects_are_equal


@dataclass
class Config:
    name: str
    value: int


config1 = Config(name="test", value=42)
config2 = Config(name="test", value=42)
objects_are_equal(config1, config2)  # Returns True
```

## Contributing Questions

### How can I contribute to `coola`?

Please read
the [contributing guide](https://github.com/durandtibo/coola/blob/main/.github/CONTRIBUTING.md) for
information on:

- Reporting bugs
- Suggesting features
- Submitting pull requests
- Development setup

### Where can I report bugs?

Please report bugs on the [GitHub issue tracker](https://github.com/durandtibo/coola/issues).

### How do I request a new feature?

Open an issue on GitHub with:

- Clear description of the feature
- Use cases and motivation
- Proposed API (if applicable)

## Other Questions

### Is `coola` actively maintained?

Yes! `coola` is actively maintained. Check
the [commit history](https://github.com/durandtibo/coola/commits/main) for recent activity.

### What license is `coola` under?

`coola` is licensed under the BSD 3-Clause "New" or "Revised" License. See
the [LICENSE](https://github.com/durandtibo/coola/blob/main/LICENSE) file for details.

### Where can I find more help?

- Documentation: https://durandtibo.github.io/coola/
- GitHub Issues: https://github.com/durandtibo/coola/issues
- GitHub Discussions: https://github.com/durandtibo/coola/discussions (if enabled)

### Can I use `coola` in commercial projects?

Yes! The BSD 3-Clause license allows commercial use. See
the [LICENSE](https://github.com/durandtibo/coola/blob/main/LICENSE) file for details.

---

**Don't see your question here?** Please open an issue
on [GitHub](https://github.com/durandtibo/coola/issues) or check
the [documentation](https://durandtibo.github.io/coola/).
