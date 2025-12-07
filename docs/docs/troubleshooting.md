# Troubleshooting

This guide helps you solve common problems when using `coola`.

## Installation Issues

### Poetry Installation Fails

**Problem**: `poetry install` fails with dependency resolution errors.

**Solution**:
```shell
# Update poetry to the latest version
poetry self update

# Clear the cache and try again
poetry cache clear pypi --all
poetry install --no-interaction
```

### Import Errors for Optional Dependencies

**Problem**: `ImportError` when trying to use PyTorch, NumPy, or other optional features.

**Solution**: Install the required optional dependencies:

```shell
# For all optional dependencies
pip install 'coola[all]'

# Or install specific dependencies
pip install coola numpy torch pandas polars xarray jax
```

### Version Conflicts

**Problem**: Conflicts with existing package versions.

**Solution**:
1. Check the [compatibility table](get_started.md#testing) for supported versions
2. Create a new virtual environment
3. Install compatible versions explicitly:

```shell
pip install 'coola==0.9.1' 'torch>=2.0,<3.0' 'numpy>=1.24,<3.0'
```

## Comparison Issues

### Unexpected Comparison Results

**Problem**: Objects that should be equal are reported as different, or vice versa.

**Diagnosis**: Use `show_difference=True` to see exactly what differs:

```python
from coola import objects_are_equal

result = objects_are_equal(obj1, obj2, show_difference=True)
```

**Common Causes**:

1. **Type mismatch**: `coola` uses strict type checking
   ```python
   objects_are_equal(1, 1.0)  # False - different types
   objects_are_equal([1, 2], (1, 2))  # False - list vs tuple
   ```

2. **Device mismatch** (PyTorch):
   ```python
   import torch
   tensor_cpu = torch.ones(2, 3)
   tensor_gpu = torch.ones(2, 3, device='cuda')
   objects_are_equal(tensor_cpu, tensor_gpu)  # False - different devices
   
   # Solution: Move to same device
   objects_are_equal(tensor_cpu, tensor_gpu.cpu())  # True
   ```

3. **Data type mismatch**:
   ```python
   import torch
   tensor_float = torch.ones(2, 3, dtype=torch.float32)
   tensor_int = torch.ones(2, 3, dtype=torch.int64)
   objects_are_equal(tensor_float, tensor_int)  # False - different dtypes
   ```

4. **NaN handling**:
   ```python
   import numpy as np
   arr1 = np.array([1.0, float('nan')])
   arr2 = np.array([1.0, float('nan')])
   objects_are_equal(arr1, arr2)  # False - NaN != NaN by default
   
   # Solution: Use equal_nan parameter
   from coola import objects_are_allclose
   objects_are_allclose(arr1, arr2, equal_nan=True)  # True
   ```

### Floating-Point Precision Issues

**Problem**: Numerical values are very close but not exactly equal.

**Solution**: Use `objects_are_allclose` with appropriate tolerance:

```python
from coola import objects_are_allclose

# Adjust tolerance as needed
result = objects_are_allclose(obj1, obj2, atol=1e-6, rtol=1e-5)
```

**Understanding tolerance**:
- `atol`: Absolute tolerance (absolute difference allowed)
- `rtol`: Relative tolerance (relative difference allowed)
- Objects are considered close if: `|a - b| <= atol + rtol * |b|`

```python
import numpy as np
from coola import objects_are_allclose

a = np.array([1.0, 1000.0])
b = np.array([1.00001, 1000.1])

# Default tolerance (atol=1e-8, rtol=1e-5)
objects_are_allclose(a, b)  # False

# Increased tolerance
objects_are_allclose(a, b, atol=0.1)  # True
```

### Dictionary Key Order Issues

**Problem**: Dictionaries with same content but different key order are equal in Python 3.7+, but you want to check order.

**Solution**: Use `OrderedDict` for order-sensitive comparisons:

```python
from collections import OrderedDict
from coola import objects_are_equal

dict1 = {"a": 1, "b": 2}
dict2 = {"b": 2, "a": 1}
objects_are_equal(dict1, dict2)  # True - dicts with same content

od1 = OrderedDict([("a", 1), ("b", 2)])
od2 = OrderedDict([("b", 2), ("a", 1)])
objects_are_equal(od1, od2)  # False - different order
```

## Performance Issues

### Slow Comparisons

**Problem**: Comparisons take too long for large objects.

**Diagnosis**:
```python
import time
from coola import objects_are_equal

start = time.time()
result = objects_are_equal(large_obj1, large_obj2)
print(f"Comparison took {time.time() - start:.2f} seconds")
```

**Solutions**:

1. **Compare specific fields** instead of entire objects:
   ```python
   # Instead of comparing entire objects
   objects_are_equal(obj1, obj2)
   
   # Compare specific fields
   objects_are_equal(obj1.data, obj2.data) and \
   objects_are_equal(obj1.metadata, obj2.metadata)
   ```

2. **Use early exit** with custom comparators:
   ```python
   # Check fast fields first
   if obj1.id != obj2.id:
       return False
   # Then check more expensive fields
   return objects_are_equal(obj1.data, obj2.data)
   ```

3. **Sample large arrays** (if exact comparison not needed):
   ```python
   import numpy as np
   from coola import objects_are_equal
   
   # For very large arrays, sample them
   sample_indices = np.random.choice(len(large_array), size=1000)
   objects_are_equal(
       large_array1[sample_indices],
       large_array2[sample_indices]
   )
   ```

### Memory Issues

**Problem**: `MemoryError` or system running out of memory during comparison.

**Solutions**:

1. **Process in chunks**:
   ```python
   def compare_in_chunks(arr1, arr2, chunk_size=1000):
       for i in range(0, len(arr1), chunk_size):
           chunk1 = arr1[i:i+chunk_size]
           chunk2 = arr2[i:i+chunk_size]
           if not objects_are_equal(chunk1, chunk2):
               return False
       return True
   ```

2. **Use memory-efficient comparisons**:
   ```python
   # Instead of creating intermediate objects
   # Compare in-place or use generators
   ```

3. **Check object size before comparison**:
   ```python
   import sys
   
   def safe_compare(obj1, obj2, max_size=100_000_000):
       size1 = sys.getsizeof(obj1)
       size2 = sys.getsizeof(obj2)
       
       if size1 > max_size or size2 > max_size:
           raise ValueError(f"Objects too large: {size1}, {size2}")
       
       return objects_are_equal(obj1, obj2)
   ```

## Type Support Issues

### Unsupported Type Error

**Problem**: `TypeError` or comparison fails for custom types.

**Error Example**:
```
TypeError: No comparison pair was able to handle inputs of type <class 'MyCustomType'>
```

**Solution**: Implement a custom comparator. See [customization guide](customization.md) for details.

**Quick Example**:
```python
from typing import Any
from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators import BaseEqualityComparator
from coola.equality.testers import EqualityTester

class MyCustomType:
    def __init__(self, value):
        self.value = value

class MyCustomComparator(BaseEqualityComparator):
    def clone(self):
        return self.__class__()
    
    def equal(self, actual: MyCustomType, expected: Any, config: EqualityConfig) -> bool:
        if type(actual) is not type(expected):
            return False
        return actual.value == expected.value

# Register the comparator
tester = EqualityTester.local_copy()
tester.add_comparator(MyCustomType, MyCustomComparator())

# Use custom tester
obj1 = MyCustomType(42)
obj2 = MyCustomType(42)
objects_are_equal(obj1, obj2, tester=tester)  # True
```

### Missing Optional Dependency

**Problem**: Comparison fails because optional package is not installed.

**Error Examples**:
```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'numpy'
```

**Solution**: Install the required package:
```shell
# Install specific packages
pip install torch
pip install numpy
pip install pandas

# Or install all optional dependencies
pip install 'coola[all]'
```

## Logging and Debugging

### Enable Detailed Logging

To see detailed comparison information:

```python
import logging

# Configure logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

from coola import objects_are_equal

# Now comparisons will show detailed logs
result = objects_are_equal(obj1, obj2, show_difference=True)
```

### Debug Nested Structures

For deeply nested structures, enable verbose logging:

```python
import logging

# Set to DEBUG for even more detail
logging.basicConfig(level=logging.DEBUG)

# Use show_difference to see the path to differences
from coola import objects_are_equal
result = objects_are_equal(
    nested_obj1,
    nested_obj2,
    show_difference=True
)
```

## Testing Issues

### Tests Fail with "Tensor-likes are not close"

**Problem**: Test fails with small numerical differences.

**Solution**: Use `objects_are_allclose` instead of `objects_are_equal`:

```python
# In tests
from coola import objects_are_allclose

def test_my_function():
    result = my_function()
    expected = get_expected_result()
    
    # Use allclose with appropriate tolerance
    assert objects_are_allclose(
        result,
        expected,
        atol=1e-6,
        rtol=1e-5,
        show_difference=True
    )
```

### Flaky Tests Due to Random Values

**Problem**: Tests sometimes pass, sometimes fail due to random initialization.

**Solution**: Set random seeds:

```python
import numpy as np
import torch
import random

def test_with_fixed_seed():
    # Fix all random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Now run your test
    result = my_function_with_random()
    expected = get_expected_result()
    assert objects_are_equal(result, expected)
```

## Common Error Messages

### "The mappings have a different value for the key X"

**Meaning**: Dictionary values differ at key X.

**Solution**: Check the value at that key in both objects.

### "The sequences have a different value at index X"

**Meaning**: List/tuple elements differ at position X.

**Solution**: Check the element at that index in both sequences.

### "torch.Tensors have different shapes"

**Meaning**: Tensors have different dimensions.

**Solution**: Check tensor shapes before comparison:
```python
print(f"Shape 1: {tensor1.shape}")
print(f"Shape 2: {tensor2.shape}")
```

### "numpy.ndarrays have different dtypes"

**Meaning**: Arrays have different data types.

**Solution**: Convert to same dtype:
```python
array2 = array2.astype(array1.dtype)
```

## Getting Help

If you can't resolve your issue:

1. **Check the documentation**: https://durandtibo.github.io/coola/
2. **Search existing issues**: https://github.com/durandtibo/coola/issues
3. **Read the FAQ**: [faq.md](faq.md)
4. **Open a new issue**: Include:
   - `coola` version: `pip show coola`
   - Python version: `python --version`
   - Operating system
   - Minimal reproducible example
   - Full error message/traceback

## Reporting Bugs

When reporting bugs, please include:

```python
# Minimal reproducible example
import coola
import sys

print(f"coola version: {coola.__version__ if hasattr(coola, '__version__') else 'unknown'}")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Your code that reproduces the issue
from coola import objects_are_equal

obj1 = ...  # Your object
obj2 = ...  # Your object

result = objects_are_equal(obj1, obj2, show_difference=True)
print(f"Result: {result}")
```

See the [contributing guide](.github/CONTRIBUTING.md) for more information on reporting issues.
