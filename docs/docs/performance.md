# Performance Guide

This guide provides tips and best practices for optimizing performance when using `coola`.

## Understanding Performance Characteristics

`coola` performance depends on several factors:

1. **Object size**: Larger objects take longer to compare
2. **Nesting depth**: Deeply nested structures require more recursive calls
3. **Data type**: Some types are faster to compare than others
4. **Comparison type**: `objects_are_equal` is generally faster than `objects_are_allclose`

## Benchmarking Your Comparisons

Before optimizing, measure your comparison performance:

```python
import time
from coola import objects_are_equal

def benchmark_comparison(obj1, obj2, iterations=100):
    start = time.time()
    for _ in range(iterations):
        result = objects_are_equal(obj1, obj2)
    elapsed = time.time() - start
    print(f"Average time: {elapsed / iterations * 1000:.2f} ms")
    return elapsed / iterations

# Example
import torch
tensor1 = torch.randn(1000, 1000)
tensor2 = torch.randn(1000, 1000)
benchmark_comparison(tensor1, tensor2)
```

## Optimization Strategies

### 1. Compare Smallest Objects First

When comparing multiple objects, check the fastest/smallest ones first:

```python
from coola import objects_are_equal

def compare_objects_optimized(obj1, obj2):
    # Check fast scalar fields first
    if obj1.id != obj2.id:
        return False
    
    if obj1.version != obj2.version:
        return False
    
    # Then check larger structures
    if not objects_are_equal(obj1.config, obj2.config):
        return False
    
    # Finally check the most expensive comparisons
    return objects_are_equal(obj1.data, obj2.data)
```

### 2. Use Appropriate Comparison Functions

Choose the right comparison function for your use case:

```python
from coola import objects_are_equal, objects_are_allclose

# For exact comparison (faster)
objects_are_equal(obj1, obj2)

# For numerical comparison with tolerance (slower)
objects_are_allclose(obj1, obj2, atol=1e-6)

# Only use allclose when you need it
if isinstance(obj1, (int, str)):
    # Use exact comparison for non-numeric types
    result = objects_are_equal(obj1, obj2)
else:
    # Use tolerance for numeric types
    result = objects_are_allclose(obj1, obj2)
```

### 3. Avoid Unnecessary Comparisons

Skip comparisons when possible:

```python
from coola import objects_are_equal

def smart_compare(obj1, obj2):
    # Quick identity check
    if obj1 is obj2:
        return True
    
    # Quick type check
    if type(obj1) is not type(obj2):
        return False
    
    # Quick size check for sequences
    if hasattr(obj1, '__len__') and len(obj1) != len(obj2):
        return False
    
    # Now do the full comparison
    return objects_are_equal(obj1, obj2)
```

### 4. Compare Metadata Before Data

For objects with both metadata and large data arrays, compare metadata first:

```python
import torch
from coola import objects_are_equal

def compare_tensors_smart(t1, t2):
    # Fast checks first
    if t1.dtype != t2.dtype:
        return False
    if t1.shape != t2.shape:
        return False
    if t1.device != t2.device:
        return False
    
    # Expensive value comparison last
    return objects_are_equal(t1, t2)
```

### 5. Use Sampling for Very Large Objects

For very large datasets where approximate equality is acceptable:

```python
import numpy as np
from coola import objects_are_equal

def compare_large_arrays_sampled(arr1, arr2, sample_size=1000):
    # Quick full checks
    if arr1.shape != arr2.shape:
        return False
    if arr1.dtype != arr2.dtype:
        return False
    
    # Sample-based comparison for large arrays
    if arr1.size > sample_size * 10:
        indices = np.random.choice(arr1.size, size=sample_size, replace=False)
        flat1 = arr1.flat[indices]
        flat2 = arr2.flat[indices]
        return objects_are_equal(flat1, flat2)
    
    # Full comparison for smaller arrays
    return objects_are_equal(arr1, arr2)
```

### 6. Process Data in Chunks

For memory-constrained environments:

```python
from coola import objects_are_equal

def compare_in_chunks(list1, list2, chunk_size=1000):
    if len(list1) != len(list2):
        return False
    
    for i in range(0, len(list1), chunk_size):
        chunk1 = list1[i:i + chunk_size]
        chunk2 = list2[i:i + chunk_size]
        
        if not objects_are_equal(chunk1, chunk2):
            return False
    
    return True
```

### 7. Disable Logging in Production

Logging can significantly impact performance:

```python
import logging
from coola import objects_are_equal

# In production, set logging to WARNING or higher
logging.getLogger('coola').setLevel(logging.WARNING)

# Don't use show_difference in performance-critical code
result = objects_are_equal(obj1, obj2, show_difference=False)
```

### 8. Cache Comparison Results

If comparing the same objects multiple times:

```python
from functools import lru_cache
from coola import objects_are_equal

@lru_cache(maxsize=128)
def cached_compare(obj1_id, obj2_id):
    # Assumes obj1 and obj2 are stored somewhere accessible
    return objects_are_equal(
        get_object_by_id(obj1_id),
        get_object_by_id(obj2_id)
    )
```

### 9. Use Native Comparison When Possible

For simple types, native comparison is faster:

```python
from coola import objects_are_equal

def optimized_compare(obj1, obj2):
    # For simple types, use native comparison
    if isinstance(obj1, (int, float, str, bool)):
        return type(obj1) == type(obj2) and obj1 == obj2
    
    # For complex types, use coola
    return objects_are_equal(obj1, obj2)
```

### 10. Parallel Comparison for Independent Objects

For comparing multiple independent pairs:

```python
from concurrent.futures import ThreadPoolExecutor
from coola import objects_are_equal

def compare_multiple_pairs(pairs):
    """Compare multiple object pairs in parallel."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda pair: objects_are_equal(pair[0], pair[1]),
            pairs
        ))
    return results

# Example usage
pairs = [
    (obj1_a, obj2_a),
    (obj1_b, obj2_b),
    (obj1_c, obj2_c),
]
results = compare_multiple_pairs(pairs)
```

## Performance Comparison: equal vs allclose

```python
import time
import torch
from coola import objects_are_equal, objects_are_allclose

def benchmark_both():
    tensor1 = torch.randn(1000, 1000)
    tensor2 = tensor1.clone()
    
    # Benchmark objects_are_equal
    start = time.time()
    for _ in range(100):
        objects_are_equal(tensor1, tensor2)
    equal_time = time.time() - start
    
    # Benchmark objects_are_allclose
    start = time.time()
    for _ in range(100):
        objects_are_allclose(tensor1, tensor2)
    allclose_time = time.time() - start
    
    print(f"objects_are_equal:    {equal_time:.4f}s")
    print(f"objects_are_allclose: {allclose_time:.4f}s")
    print(f"Speedup: {allclose_time / equal_time:.2f}x")

benchmark_both()
```

## Memory Optimization

### 1. Avoid Creating Intermediate Objects

```python
# Less efficient - creates intermediate objects
def compare_inefficient(data1, data2):
    processed1 = [x * 2 for x in data1]
    processed2 = [x * 2 for x in data2]
    return objects_are_equal(processed1, processed2)

# More efficient - compare directly
def compare_efficient(data1, data2):
    return objects_are_equal(data1, data2)
```

### 2. Use Generators for Large Sequences

```python
from coola import objects_are_equal

# For very large sequences, compare elements one by one
def compare_generators(gen1, gen2):
    for item1, item2 in zip(gen1, gen2):
        if not objects_are_equal(item1, item2):
            return False
    return True
```

### 3. Release Memory After Comparison

```python
import gc
from coola import objects_are_equal

def compare_and_cleanup(obj1, obj2):
    result = objects_are_equal(obj1, obj2)
    
    # Release references
    del obj1, obj2
    
    # Force garbage collection if needed
    gc.collect()
    
    return result
```

## Real-World Performance Tips

### Tip 1: Profile Your Code

Use Python profilers to identify bottlenecks:

```python
import cProfile
import pstats
from coola import objects_are_equal

def profile_comparison(obj1, obj2):
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = objects_are_equal(obj1, obj2)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Print top 10 functions
    
    return result
```

### Tip 2: Monitor Memory Usage

```python
import tracemalloc
from coola import objects_are_equal

def measure_memory(obj1, obj2):
    tracemalloc.start()
    
    result = objects_are_equal(obj1, obj2)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    return result
```

### Tip 3: Use Type-Specific Optimizations

For known types, use specialized comparison logic:

```python
import torch
import numpy as np
from coola import objects_are_equal

def compare_known_types(obj1, obj2):
    # PyTorch tensors
    if isinstance(obj1, torch.Tensor) and isinstance(obj2, torch.Tensor):
        return (obj1.dtype == obj2.dtype and 
                obj1.device == obj2.device and
                obj1.shape == obj2.shape and
                torch.equal(obj1, obj2))
    
    # NumPy arrays
    elif isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        return (obj1.dtype == obj2.dtype and
                obj1.shape == obj2.shape and
                np.array_equal(obj1, obj2))
    
    # Fall back to coola for other types
    else:
        return objects_are_equal(obj1, obj2)
```

## Performance Checklist

Before deploying comparison-heavy code:

- [ ] Benchmark your specific use case
- [ ] Use the right comparison function (`equal` vs `allclose`)
- [ ] Compare fast/small objects first
- [ ] Disable detailed logging in production
- [ ] Consider sampling for very large objects
- [ ] Profile to identify bottlenecks
- [ ] Monitor memory usage
- [ ] Cache results when appropriate
- [ ] Use parallel comparison for independent pairs
- [ ] Implement early-exit strategies

## Expected Performance

As a rough guideline (performance varies by hardware and object size):

| Object Type | Size | Approximate Time |
|-------------|------|------------------|
| Simple scalars | - | < 1 μs |
| Small tensors | 10x10 | ~10 μs |
| Medium tensors | 1000x1000 | ~1 ms |
| Large tensors | 10000x10000 | ~100 ms |
| Nested dict (depth 3) | 100 items | ~100 μs |
| Deep nesting (depth 10) | 1000 items | ~10 ms |

These are rough estimates and will vary significantly based on:
- Hardware (CPU speed, memory)
- Data types and values
- Nesting structure
- Python version and optimizations

## When Performance Matters

Focus on optimization when:

1. **High-frequency comparisons**: Comparing objects thousands of times per second
2. **Large objects**: Comparing multi-GB datasets
3. **Real-time systems**: Latency requirements < 100ms
4. **Resource-constrained environments**: Limited memory or CPU

For most use cases, `coola`'s default behavior is sufficient.

## Getting Help

If you're experiencing performance issues:

1. Profile your code to identify bottlenecks
2. Check the [troubleshooting guide](troubleshooting.md)
3. Open an issue on [GitHub](https://github.com/durandtibo/coola/issues) with:
   - Performance profile output
   - Object sizes and types
   - Expected vs actual performance
