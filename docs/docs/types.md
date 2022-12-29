# Supported Types

:book: This page describes what types are currently supported and what rules are used to check if
two objects are equal or not.

## Equal

TODO: add default

### `collections.abc.Mapping` / `dict`

Two `Mapping`s are equal if:

- they have the same type
- they have the same number of elements i.e. `len(mapping1) == len(mapping2)` returns `True`
- they have the same set of keys i.e. `set(mapping1.keys()) != set(mapping2.keys())` returns `True`
- For each key, the values are equal. The value associated to the key `k` in the first mapping has
  to be equal to value associated to the key `k` in the second mapping.

```python
from collections import OrderedDict

from coola import objects_are_equal

objects_are_equal({'int': 1, 'str': 'abc'}, {'int': 1, 'str': 'abc'})  # True
objects_are_equal({'int': 1, 'str': 'abc'}, OrderedDict({'int': 1, 'str': 'abc'}))  # False
objects_are_equal({'int': 1, 'str': 'abc'}, {'int': 1, 'str': 'abc', 'float': 0.2})  # False
objects_are_equal({'int': 1, 'str': 'abc'}, {'int': 1, 'float': 0.2})  # False
objects_are_equal({'int': 1, 'str': 'abc'}, {'int': 1, 'str': 'abcd'})  # False
```

### `collections.abc.Sequence` / `list` / `tuple`

Two `Sequence`s are equal if:

- they have the same type
- they have the same number of elements i.e. `len(sequence1) == len(sequence2)` returns `True`
- For each position, the elements are equal. The `i`-th element in the first sequence has to be
  equal to the `i`-th element in the second sequence.

**Example**

```python
from coola import objects_are_equal

objects_are_equal([1, 2, 'abc'], [1, 2, 'abc'])  # True
objects_are_equal([1, 2, 'abc'], (1, 2, 'abc'))  # False
objects_are_equal([1, 2, 'abc'], [1, 2, 'abc', 4])  # False
objects_are_equal([1, 2, 'abc'], [1, 2, 'abcd'])  # False
```

### PyTorch

You need to install `coola` with PyTorch to check if some PyTorch objects are equal or not. Please
check the [get started page](get_started.md) for more information. `coola` currently support the
following PyTorch objects:

- `torch.Tensor`
- `torch.nn.utils.rnn.PackedSequence`

#### `torch.Tensor`

Two PyTorch `Tensor`s are equal if:

- they have the same data type i.e. `tensor1.dtype == tensor2.dtype` returns `True`
- they have the same device i.e. `tensor1.device == tensor2.device` returns `True`
- they have the same shape i.e. `tensor1.shape == tensor2.shape` returns `True`
- they have the same values i.e. `tensor1.equal(tensor2)` returns `True`

**Example**

```python
import torch

from coola import objects_are_equal

objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))  # True
objects_are_equal(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))  # False
objects_are_equal(torch.ones(2, 3), torch.ones(2, 3, device='cuda'))  # False
objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3))  # False
objects_are_equal(torch.ones(2, 3), torch.ones(6))  # False
```

#### `torch.nn.utils.rnn.PackedSequence`

Two PyTorch `PackedSequence`s are equal if:

- The `data` attributes are equal
- The `batch_sizes` attributes are equal
- The `sorted_indices` attributes are equal
- The `unsorted_indices` attributes are equal

**Example**

```python
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from coola import objects_are_equal

objects_are_equal(
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
)  # True

objects_are_equal(
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).add(1).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
)  # False | different values

objects_are_equal(
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 2], dtype=torch.long),
    batch_first=True,
  ),
)  # False | different lengths
```

### `numpy.ndarray`

You need to install `coola` with NumPy to check if some NumPy ndarrays are equal or not. Please
check the [get started page](get_started.md) for more information.

Two NumPy `ndarray`s are equal if:

- they have the same data type i.e. `array1.dtype == array2.dtype` returns `True`
- they have the same shape i.e. `array1.shape == array2.shape` returns `True`
- they have the same values i.e. `numpy.array_equal(array1, array2)` returns `True`

```python
import numpy

from coola import objects_are_equal

objects_are_equal(numpy.ones((2, 3)), numpy.ones((2, 3)))  # True
objects_are_equal(numpy.ones((2, 3)), numpy.ones((2, 3), dtype=int))  # False
objects_are_equal(numpy.ones((2, 3)), numpy.zeros((2, 3)))  # False
objects_are_equal(numpy.ones((2, 3)), numpy.ones((6,)))  # False
```

## Equal within a tolerance (allclose)

### `collections.abc.Mapping` / `dict`

Two `Mapping`s are equal within a tolerance if:

- they have the same type
- they have the same number of elements i.e. `len(mapping1) == len(mapping2)` returns `True`
- they have the same set of keys i.e. `set(mapping1.keys()) != set(mapping2.keys())` returns `True`
- For each key, the values are equal within a tolerance. The value associated to the key `k` in the
  first mapping has to be equal within the tolerance to value associated to the key `k` in the
  second mapping.

```python
from collections import OrderedDict

from coola import objects_are_allclose

objects_are_allclose({'int': 1, 'str': 'abc'}, {'int': 1, 'str': 'abc'})  # True
objects_are_allclose({'int': 1, 'str': 'abc'}, {'int': 2, 'str': 'abc'}, atol=2)  # True
objects_are_allclose({'int': 1, 'str': 'abc'}, {'int': 2, 'str': 'abc'}, rtol=1)  # True
objects_are_allclose({'int': 1, 'str': 'abc'}, OrderedDict({'int': 1, 'str': 'abc'}))  # False
objects_are_allclose({'int': 1, 'str': 'abc'}, {'int': 1, 'str': 'abc', 'float': 0.2})  # False
objects_are_allclose({'int': 1, 'str': 'abc'}, {'int': 1, 'float': 0.2})  # False
objects_are_allclose({'int': 1, 'str': 'abc'}, {'int': 1, 'str': 'abcd'})  # False
```

### `collections.abc.Sequence` / `list` / `tuple`

Two `Sequence`s are equal within a tolerance if:

- they have the same type
- they have the same number of elements i.e. `len(sequence1) == len(sequence2)` returns `True`
- For each position, the elements are equal within a tolerance. The `i`-th element in the first
  sequence has to be equal within a tolerance to the `i`-th element in the second sequence.

**Example**

```python
from coola import objects_are_allclose

objects_are_allclose([1, 2, 'abc'], [1, 2, 'abc'])  # True
objects_are_allclose([1, 2, 'abc'], [1, 3, 'abc'], atol=2)  # True
objects_are_allclose([1, 2, 'abc'], [1, 3, 'abc'], rtol=1)  # True
objects_are_allclose([1, 2, 'abc'], (1, 2, 'abc'))  # False
objects_are_allclose([1, 2, 'abc'], [1, 2, 'abc', 4])  # False
objects_are_allclose([1, 2, 'abc'], [1, 2, 'abcd'])  # False
```

### PyTorch

You need to install `coola` with PyTorch to check if some PyTorch objects are equal within a
tolerance or not. Please check the [get started page](get_started.md) for more information. `coola`
currently support the following PyTorch objects:

- `torch.Tensor`
- `torch.nn.utils.rnn.PackedSequence`

#### `torch.Tensor`

Two PyTorch `Tensor`s are equal if:

- they have the same data type i.e. `tensor1.dtype == tensor2.dtype` returns `True`
- they have the same device i.e. `tensor1.device == tensor2.device` returns `True`
- they have the same shape i.e. `tensor1.shape == tensor2.shape` returns `True`
- the values are equal within a tolerance i.e. `tensor1.allclose(tensor2)` returns `True`

**Example**

```python
import torch

from coola import objects_are_allclose

objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3))  # True
objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3) + 1, atol=2)  # True
objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3) + 1, rtol=1)  # True
objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))  # False
objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3, device='cuda'))  # False
objects_are_allclose(torch.ones(2, 3), torch.zeros(2, 3))  # False
objects_are_allclose(torch.ones(2, 3), torch.ones(6))  # False
```

#### `torch.nn.utils.rnn.PackedSequence`

Two PyTorch `PackedSequence`s are equal within a tolerance if:

- The `data` attributes are equal within a tolerance
- The `batch_sizes` attributes are equal
- The `sorted_indices` attributes are equal
- The `unsorted_indices` attributes are equal

**Example**

```python
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from coola import objects_are_allclose

objects_are_allclose(
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
)  # True

objects_are_allclose(
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float() + 1,
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  atol=2,
)  # True

objects_are_allclose(
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).add(1).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
)  # False | different values

objects_are_allclose(
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 3], dtype=torch.long),
    batch_first=True,
  ),
  pack_padded_sequence(
    input=torch.arange(10).view(2, 5).float(),
    lengths=torch.tensor([5, 2], dtype=torch.long),
    batch_first=True,
  ),
)  # False | different lengths
```

### `numpy.ndarray`

You need to install `coola` with NumPy to check if some NumPy ndarrays are equal or not. Please
check the [get started page](get_started.md) for more information.

Two NumPy `ndarray`s are equal within a tolerance if:

- they have the same data type i.e. `array1.dtype == array2.dtype` returns `True`
- they have the same shape i.e. `array1.shape == array2.shape` returns `True`
- the values are equal within a tolerance i.e. `numpy.allclose(array1, array2)` returns `True`

```python
import numpy

from coola import objects_are_allclose

objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)))  # True
objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)) + 1, atol=2)  # True
objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)) + 1, rtol=1)  # True
objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3), dtype=int))  # False
objects_are_allclose(numpy.ones((2, 3)), numpy.zeros((2, 3)))  # False
objects_are_allclose(numpy.ones((2, 3)), numpy.ones((6,)))  # False
```
