# Supported Types

:book: This page describes what types are currently supported and what rules are used to check if
two objects are equal or not.

The current supported types are:

- [`jax.numpy.ndarray`](https://jax.readthedocs.io/en/latest/index.html)
- [`numpy.ndarray`](https://numpy.org/doc/stable/index.html)
- [`numpy.ma.MaskedArray`](https://numpy.org/doc/stable/reference/maskedarray.generic.html)
- [`pandas.DataFrame`](https://pandas.pydata.org/)
- [`pandas.Series`](https://pandas.pydata.org/)
- [`polars.DataFrame`](https://www.pola.rs/)
- [`polars.Series`](https://www.pola.rs/)
- [`torch.Tensor`](https://pytorch.org/)
- [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/)
- [`xarray.DataArray`](https://docs.xarray.dev/en/stable/)
- [`xarray.Dataset`](https://docs.xarray.dev/en/stable/)
- [`xarray.Variable`](https://docs.xarray.dev/en/stable/)

## Equal

### `object`

By default, two objects are equal if:

- they have the same type
- they are equal i.e. `actual == expected` returns `True`

**Example**

```pycon
>>> from coola import objects_are_equal
>>> objects_are_equal(1, 1)
True
>>> objects_are_equal(1, 2)
False
>>> objects_are_equal(1, 1.0)
False
>>> objects_are_equal(True, True)
True
>>> objects_are_equal("abc", "abcd")
False

```

### `collections.abc.Mapping` | `dict`

Two `Mapping`s are equal if:

- they have the same type
- they have the same number of elements i.e. `len(mapping1) == len(mapping2)` returns `True`
- they have the same set of keys i.e. `set(mapping1.keys()) != set(mapping2.keys())` returns `True`
- For each key, the values are equal. The value associated to the key `k` in the first mapping has
  to be equal to value associated to the key `k` in the second mapping.

```pycon
>>> from collections import OrderedDict
>>> from coola import objects_are_equal
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "str": "abc"})
True
>>> objects_are_equal({"int": 1, "str": "abc"}, OrderedDict({"int": 1, "str": "abc"}))
False
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "str": "abc", "float": 0.2})
False
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "float": 0.2})
False
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "str": "abcd"})
False

```

### `collections.abc.Sequence` | `list` | `tuple`

Two `Sequence`s are equal if:

- they have the same type
- they have the same number of elements i.e. `len(sequence1) == len(sequence2)` returns `True`
- For each position, the elements are equal. The `i`-th element in the first sequence has to be
  equal to the `i`-th element in the second sequence.

**Example**

```pycon
>>> from coola import objects_are_equal
>>> objects_are_equal([1, 2, "abc"], [1, 2, "abc"])
True
>>> objects_are_equal([1, 2, "abc"], (1, 2, "abc"))
False
>>> objects_are_equal([1, 2, "abc"], [1, 2, "abc", 4])
False
>>> objects_are_equal([1, 2, "abc"], [1, 2, "abcd"])
False

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

```pycon
>>> import torch
>>> from coola import objects_are_equal
>>> objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))
True
>>> objects_are_equal(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))
False
>>> objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3))
False
>>> objects_are_equal(torch.ones(2, 3), torch.ones(6))
False

```

#### `torch.nn.utils.rnn.PackedSequence`

Two PyTorch `PackedSequence`s are equal if:

- The `data` attributes are equal
- The `batch_sizes` attributes are equal
- The `sorted_indices` attributes are equal
- The `unsorted_indices` attributes are equal

**Example**

```pycon
>>> import torch
>>> from torch.nn.utils.rnn import pack_padded_sequence
>>> from coola import objects_are_equal
>>> objects_are_equal(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )
True
>>> objects_are_equal(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).add(1).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different values
False
>>> objects_are_equal(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 2], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different lengths
False

```

### `numpy.ndarray`

You need to install `coola` with NumPy to check if some NumPy ndarrays are equal or not. Please
check the [get started page](get_started.md) for more information.

Two NumPy `ndarray`s are equal if:

- they have the same data type i.e. `array1.dtype == array2.dtype` returns `True`
- they have the same shape i.e. `array1.shape == array2.shape` returns `True`
- they have the same values i.e. `numpy.array_equal(array1, array2)` returns `True`

```pycon
>>> import numpy
>>> from coola import objects_are_equal
>>> objects_are_equal(numpy.ones((2, 3)), numpy.ones((2, 3)))
True
>>> objects_are_equal(numpy.ones((2, 3)), numpy.ones((2, 3), dtype=int))
False
>>> objects_are_equal(numpy.ones((2, 3)), numpy.zeros((2, 3)))
False
>>> objects_are_equal(numpy.ones((2, 3)), numpy.ones((6,)))
False

```

### xarray

You need to install `coola` with PyTorch to check if some xarray objects are equal or not. Please
check the [get started page](get_started.md) for more information. `coola` currently support the
following xarray objects:

- `xarray.DataArray`
- `xarray.Dataset`

#### `xarray.DataArray`

Two xarray `DataArray`s are equal if:

- they have the same data values (attribute `data`)
- they have the same name (attribute `name`)
- they have the same dimension names (attribute `dims`)
- they have the same coordinates (attribute `coords`)
- they have the same metadata (attribute `attrs`)

Unlike `xarray.DataArray.identical`, two `DataArray`s are not equal if both objects have NaNs in the
same positions to follow the standard usage in numpy.
You can use `objects_are_allclose` to compare objects with NaNs.

```pycon
>>> import numpy as np
>>> import xarray as xr
>>> from coola import objects_are_equal
>>> objects_are_equal(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"]))
True
>>> objects_are_equal(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.zeros(6), dims=["z"]))
False

```

#### `xarray.Dataset`

Two xarray `Dataset`s are equal if `xarray.Dataset.identical` returns `True`.
In contrast to the standard usage in numpy, NaNs are compared like numbers, two `Dataset`s are equal
if both objects have NaNs in the same positions.

```pycon
>>> import numpy as np
>>> import xarray as xr
>>> from coola import objects_are_equal
>>> ds1 = xr.Dataset(
...     {
...         "x": xr.DataArray(
...             np.arange(6),
...             dims=["z"],
...         ),
...         "y": xr.DataArray(
...             np.ones((6, 3)),
...             dims=["z", "t"],
...         ),
...     },
...     coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
... )
>>> ds2 = xr.Dataset(
...     {
...         "x": xr.DataArray(
...             np.arange(6),
...             dims=["z"],
...         ),
...         "y": xr.DataArray(
...             np.ones((6, 3)),
...             dims=["z", "t"],
...         ),
...     },
...     coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
... )
>>> ds3 = xr.Dataset(
...     {
...         "x": xr.DataArray(
...             np.arange(6),
...             dims=["z"],
...         ),
...     },
...     coords={"z": np.arange(6) + 1},
... )
>>> objects_are_equal(ds1, ds2)
True
>>> objects_are_equal(ds1, ds3)
False

```

## Equal within a tolerance (allclose)

### `object`

The concept of equal within a tolerance does not make sense for all `object`s.
In general, the tolerance is not used for `object`s.
The tolerance is only used for numbers (see below).
By default, two objects are equal if:

- they have the same type
- they are equal i.e. `actual == expected` returns `True`

**Example**

```pycon
>>> from coola import objects_are_allclose
>>> objects_are_allclose("abc", "abc")
True
>>> objects_are_allclose("abc", "abcd")
False

```

### Numbers: `bool` | `int` | `float`

Two numbers are equal within a tolerance if:

- they have the same type
- the values are equal with a tolerance

**Example**

```pycon
>>> from coola import objects_are_allclose
>>> objects_are_allclose(1, 2)
False
>>> objects_are_allclose(1, 2, atol=1)
True
>>> objects_are_allclose(1, 2, rtol=1)
True
>>> objects_are_allclose(1.0, 2.0)
False
>>> objects_are_allclose(1.0, 2.0, atol=1)
True
>>> objects_are_allclose(1.0, 2.0, rtol=1)
True
>>> objects_are_allclose(1, 2.0, atol=1)
False

```

Note that booleans are explicitly considered as integers in Python so the tolerance can be used with
booleans:

```pycon
>>> from coola import objects_are_allclose
>>> objects_are_allclose(True, False)
False
>>> objects_are_allclose(True, False, atol=1)
True
>>> objects_are_allclose(True, False, rtol=1)
True

```

### `collections.abc.Mapping` | `dict`

Two `Mapping`s are equal within a tolerance if:

- they have the same type
- they have the same number of elements i.e. `len(mapping1) == len(mapping2)` returns `True`
- they have the same set of keys i.e. `set(mapping1.keys()) != set(mapping2.keys())` returns `True`
- For each key, the values are equal within a tolerance. The value associated to the key `k` in the
  first mapping has to be equal within the tolerance to value associated to the key `k` in the
  second mapping.

**Example**

```pycon
>>> from collections import OrderedDict
>>> from coola import objects_are_allclose
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "str": "abc"})
True
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 2, "str": "abc"}, atol=2)
True
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 2, "str": "abc"}, rtol=1)
True
>>> objects_are_allclose({"int": 1, "str": "abc"}, OrderedDict({"int": 1, "str": "abc"}))
False
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "str": "abc", "float": 0.2})
False
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "float": 0.2})
False
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "str": "abcd"})
False

```

### `collections.abc.Sequence` | `list` | `tuple`

Two `Sequence`s are equal within a tolerance if:

- they have the same type
- they have the same number of elements i.e. `len(sequence1) == len(sequence2)` returns `True`
- For each position, the elements are equal within a tolerance. The `i`-th element in the first
  sequence has to be equal within a tolerance to the `i`-th element in the second sequence.

**Example**

```pycon
>>> from coola import objects_are_allclose
>>> objects_are_allclose([1, 2, "abc"], [1, 2, "abc"])
True
>>> objects_are_allclose([1, 2, "abc"], [1, 3, "abc"], atol=2)
True
>>> objects_are_allclose([1, 2, "abc"], [1, 3, "abc"], rtol=1)
True
>>> objects_are_allclose([1, 2, "abc"], (1, 2, "abc"))
False
>>> objects_are_allclose([1, 2, "abc"], [1, 2, "abc", 4])
False
>>> objects_are_allclose([1, 2, "abc"], [1, 2, "abcd"])
False

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

```pycon
>>> import torch
>>> from coola import objects_are_allclose
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3))
True
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3) + 1, atol=2)
True
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3) + 1, rtol=1)
True
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))
False
>>> objects_are_allclose(torch.ones(2, 3), torch.zeros(2, 3))
False
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(6))
False

```

#### `torch.nn.utils.rnn.PackedSequence`

Two PyTorch `PackedSequence`s are equal within a tolerance if:

- The `data` attributes are equal within a tolerance
- The `batch_sizes` attributes are equal
- The `sorted_indices` attributes are equal
- The `unsorted_indices` attributes are equal

**Example**

```pycon
>>> import torch
>>> from torch.nn.utils.rnn import pack_padded_sequence
>>> from coola import objects_are_allclose
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )
True
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float() + 1,
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     atol=2,
... )
True
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).add(1).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different values
False
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 2], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different lengths
False

```

### `numpy.ndarray`

You need to install `coola` with NumPy to check if some NumPy ndarrays are equal or not. Please
check the [get started page](get_started.md) for more information.

Two NumPy `ndarray`s are equal within a tolerance if:

- they have the same data type i.e. `array1.dtype == array2.dtype` returns `True`
- they have the same shape i.e. `array1.shape == array2.shape` returns `True`
- the values are equal within a tolerance i.e. `numpy.allclose(array1, array2)` returns `True`

```pycon
>>> import numpy
>>> from coola import objects_are_allclose
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)))
True
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)) + 1, atol=2)
True
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)) + 1, rtol=1)
True
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3), dtype=int))
False
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.zeros((2, 3)))
False
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((6,)))
False

```

### xarray

You need to install `coola` with PyTorch to check if some xarray objects are equal or not. Please
check the [get started page](get_started.md) for more information. `coola` currently support the
following xarray objects:

- `xarray.DataArray`
- `xarray.Dataset`

#### `xarray.DataArray`

Two xarray `DataArray`s are equal within a tolerance if:

- they have the same data values within the tolerance (attribute `data`)
- they have the same name (attribute `name`)
- they have the same dimension names (attribute `dims`)
- they have the same coordinates (attribute `coords`)
- they have the same metadata (attribute `attrs`)

Unlike `xarray.DataArray.identical`, two `DataArray`s are not equal if both objects have NaNs in the
same positions to follow the standard usage in numpy.
You can use `objects_are_allclose` to compare objects with NaNs.

```pycon
>>> import numpy as np
>>> import xarray as xr
>>> from coola import objects_are_allclose
>>> objects_are_allclose(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"]))
True
>>> objects_are_allclose(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.zeros(6), dims=["z"]))
False

```
