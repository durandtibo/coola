# coola quickstart

:book: This page is a quick overview of the two main functions of `coola`: `objects_are_equal`
and `objects_are_allclose`.
These functions can be used to check if two complex/nested objects are equal or not.
The motivation of the library is explained [here](index.md#motivation).
You should read this page if you want to learn how to use these functions.
This page does not explain the internal behavior of these functions.

**Prerequisites:** You’ll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).
It is highly recommended to know a bit of [NumPy](https://numpy.org/doc/stable/user/quickstart.html)
or [PyTorch](https://pytorch.org/tutorials/).

## Equal or not?

`coola` provides a function `objects_are_equal` that can indicate if two complex/nested objects are
equal or not. It also works for simple objects like integer or string.

### First example

The following example shows how to use the `objects_are_equal` function.
The objects to compare are dictionaries containing a PyTorch `Tensor` and a NumPy `ndarray`.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> data3 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> objects_are_equal(data1, data2)
False
>>> objects_are_equal(data1, data3)
True

```

In one line, it is possible to check two complex/nested objects are equal or not.
Unlike the native python equality operator `==`, the `objects_are_equal` function can check if two
dictionaries containing PyTorch `Tensor`s and NumPy `ndarray`s are equal or not.

### Finding a difference

When the objects are complex or nested, it is not obvious to know which elements are different.
This function has an argument `show_difference` which shows the first difference found between the
two objects. For example if you add `show_difference=True` when you compare the `data1`
and `data2`, you will see at least one element that is different:

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> objects_are_equal(data1, data2, show_difference=True)
False

```

*Log output*:

```textmate
INFO:coola.comparators.torch_:torch.Tensors are different
actual=
tensor([[1., 1., 1.],
        [1., 1., 1.]])
expected=
tensor([[0., 0., 0.],
        [0., 0., 0.]])
INFO:coola.comparators.equality:The mappings have a different value for the key 'torch':
first mapping  = {'torch': tensor([[1., 1., 1.],
        [1., 1., 1.]]), 'numpy': array([[0., 0., 0.],
       [0., 0., 0.]])}
second mapping = {'torch': tensor([[0., 0., 0.],
        [0., 0., 0.]]), 'numpy': array([[1., 1., 1.],
       [1., 1., 1.]])}
```

If you do not see this output, you may need to configure `logging` to show the `INFO`
level (something like `logging.basicConfig(level=logging.INFO)`). The log shows a difference
between `data1` and `data2`: the PyTorch `Tensor`s in key `'torch'` of the input dictionaries.
The top of the log shows the element that fails the check, and then it shows the parent element, so
it is easy to know where is the identified difference.
Note that it only shows the first difference, not all the differences. Two objects are different if
any of these elements are different. In the previous example, only the difference for key `'torch'`
is shown in the log.
No log is shown if the two objects are equal and `show_difference=True`.

### More examples

The previous examples use dictionary, but it is possible to use other types like list or tuple

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_equal
>>> data1 = [torch.ones(2, 3), numpy.zeros((2, 3))]
>>> data2 = [torch.zeros(2, 3), numpy.ones((2, 3))]
>>> data3 = (torch.ones(2, 3), numpy.zeros((2, 3)))
>>> objects_are_equal(data1, data2)
False
>>> objects_are_equal(data1, data3)
False

```

It is also possible to test more complex objects

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_equal
>>> data1 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3))],
...     "dict": {"torch": torch.arange(5), "str": "abc"},
...     "int": 1,
... }
>>> data2 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3))],
...     "dict": {"torch": torch.arange(5), "str": "abcd"},
...     "int": 1,
... }
...
>>> objects_are_equal(data1, data2)
False

```

Feel free to try any complex nested structure that you want. You can find the currently supported
types [here](types.md#equal).

### Strict type checking

:warning: Unlike the native python equality operator `==`, the `objects_are_equal` function requires
two
objects to be of the same type to be equal.
For example, `1` (integer) is considered different from `1.0` (float) or `True` (boolean) which is
different behavior that the native python equality operator `==`. You can take a look to the
following example to see some differences.

```pycon

>>> from coola import objects_are_equal
>>> objects_are_equal(1, 1)
True
>>> objects_are_equal(1, 1.0)
False
>>> objects_are_equal(1, True)
False
>>> 1 == 1
True
>>> 1 == 1.0
True
>>> 1 == True
True

```

Similarly, the `objects_are_equal` function considers a `dict` and `collections.OrderedDict` as
different objects even if they have the same keys and values.

```pycon

>>> from collections import OrderedDict
>>> from coola import objects_are_equal
>>> objects_are_equal({"key1": 1, "key2": "abc"}, OrderedDict({"key1": 1, "key2": "abc"}))
False
>>> {"key1": 1, "key2": "abc"} == OrderedDict({"key1": 1, "key2": "abc"})
True

```

## Almost equal or not?

`coola` provides a function `objects_are_allclose` that can indicate if two complex/nested objects
are equal within a tolerance or not.
Due to numerical precision, it happens quite often that two numbers are not equal but the error is
very tiny (`1.0` and `1.000000001`).
The tolerance is mostly useful for numerical values.
For a lot of types like string, the `objects_are_allclose` function behaves like
the `objects_are_equal` function.

### First example

The following example shows how to use the `objects_are_allclose` function.
The objects to compare are dictionaries containing a PyTorch Tensor and a NumPy ndarray.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_allclose, objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> data3 = {"torch": torch.ones(2, 3) + 1e-9, "numpy": numpy.zeros((2, 3)) - 1e-9}
>>> objects_are_allclose(data1, data2)
False
>>> objects_are_allclose(data1, data3)
True
>>> objects_are_equal(data1, data3)
False

```

The difference between `data1` and `data2` is large so `objects_are_allclose` returns false
like `objects_are_equal`. The difference between `data1` and `data3` is tiny
so `objects_are_allclose` returns true, whereas `objects_are_equal` returns false.

### Tolerance

It is possible to control the tolerance with the arguments `atol` and `rtol`. `atol` controls the
absolute tolerance and `rtol` controls the relative tolerance.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_allclose
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.ones(2, 3) + 1e-4, "numpy": numpy.zeros((2, 3)) - 1e-4}
>>> objects_are_allclose(data1, data2)
False
>>> objects_are_allclose(data1, data2, atol=1e-3)
True

```

`objects_are_equal` and `objects_are_allclose` are very similar and should behave the same
when `atol=0.0` and `rtol=0.0`.

### Finding a difference

Like `objects_are_equal`, the `objects_are_allclose` function has an argument `show_difference`
which shows the first difference found between the two objects.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_allclose
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.ones(2, 3) + 1e-4, "numpy": numpy.zeros((2, 3)) - 1e-4}
>>> objects_are_allclose(data1, data2, show_difference=True)
False

```

*Output*:

```textmate
INFO:coola.comparators.torch_:torch.Tensors are different
actual=
tensor([[1., 1., 1.],
        [1., 1., 1.]])
expected=
tensor([[1.0001, 1.0001, 1.0001],
        [1.0001, 1.0001, 1.0001]])
INFO:coola.comparators.allclose:The mappings have a different value for the key torch:
first mapping  = {'torch': tensor([[1., 1., 1.],
        [1., 1., 1.]]), 'numpy': array([[0., 0., 0.],
       [0., 0., 0.]])}
second mapping = {'torch': tensor([[1.0001, 1.0001, 1.0001],
        [1.0001, 1.0001, 1.0001]]), 'numpy': array([[-0.0001, -0.0001, -0.0001],
       [-0.0001, -0.0001, -0.0001]])}
```

### More examples

Like the `objects_are_equal` function, the `objects_are_allclose` function can be used with
complex/nested objects.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_allclose
>>> data1 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3))],
...     "dict": {"torch": torch.arange(5), "str": "abc"},
...     "int": 1,
... }
>>> data2 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3)) + 1e-9],
...     "dict": {"torch": torch.arange(5), "str": "abc"},
...     "int": 1,
... }
>>> objects_are_allclose(data1, data2)
True

```

`objects_are_allclose` supports a lot of types and nested structure.
Feel free to try any complex nested structure that you want. You can find the currently supported
types [here](types.md#equal-within-a-tolerance--allclose-).

### Not A Number (NaN)

By default, `NaN` is not considered close to any other value, including `NaN`.

```pycon

>>> from coola import objects_are_allclose
>>> objects_are_allclose(float("nan"), 0.0)
False
>>> objects_are_allclose(float("nan"), float("nan"))
False

```

By setting `equal_nan=True`, it is possible to change the above behavior and `NaN`s will be
considered equal.

```pycon

>>> from coola import objects_are_allclose
>>> objects_are_allclose(float("nan"), float("nan"), equal_nan=True)
True

```

In arrays or tensors, `NaN` are sometimes to indicate some values are not valid.
However, it may be interested to check if the non-`NaN` values are equal.
It is possible to use the `equal_nan=True` option to compare two tensors with `NaN` values.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_allclose
>>> objects_are_allclose(
...     torch.tensor([0.0, 1.0, float("nan")]),
...     torch.tensor([0.0, 1.0, float("nan")]),
... )
False
>>> objects_are_allclose(
...     torch.tensor([0.0, 1.0, float("nan")]),
...     torch.tensor([0.0, 1.0, float("nan")]),
...     equal_nan=True,
... )
True
>>> objects_are_allclose(
...     numpy.array([0.0, 1.0, float("nan")]),
...     numpy.array([0.0, 1.0, float("nan")]),
... )
False
>>> objects_are_allclose(
...     numpy.array([0.0, 1.0, float("nan")]),
...     numpy.array([0.0, 1.0, float("nan")]),
...     equal_nan=True,
... )
True

```

*Output*:

```textmate
False
True
False
True
```

## Connection with similar tools

`coola` is not the first tool to provide functions to compare nested objects.
If you are a PyTorch user, you probably know
the [`torch.testing.assert_close`](https://pytorch.org/docs/stable/testing.html) function.
If you are a NumPy user, you probably know
the [`numpy.testing.assert_equal`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_equal.html)
function.
However, most of these functions work in a fix scope and are difficult to extend or customize.
On the opposite side, `coola` is flexible and easy to customize.

Let's take a look to `torch`.
[`torch.testing.assert_close`](https://pytorch.org/docs/stable/testing.html) allows to easily
compare `torch.Tensor`s objects:

```pycon

>>> import torch
>>> torch.testing.assert_close(torch.ones(2, 3), torch.ones(2, 3))

```

It can also be used on mappings or sequences:

```pycon

>>> import torch
>>> torch.testing.assert_close(
...     [torch.ones(2, 3), torch.zeros(3)],
...     [torch.ones(2, 3), torch.zeros(3)],
... )
>>> torch.testing.assert_close(
...     {'key1': torch.ones(2, 3), 'key2': torch.zeros(3)},
...     {'key1': torch.ones(2, 3), 'key2': torch.zeros(3)},
... )
>>> torch.testing.assert_close(
...     {'key1': torch.ones(2, 3), 'key2': {'key3': torch.zeros(3), 'key4': [torch.arange(5)]}},
...     {'key1': torch.ones(2, 3), 'key2': {'key3': torch.zeros(3), 'key4': [torch.arange(5)]}},
... )

```

It also works on tensor like objects like NumPy arrays:

```pycon

>>> import torch
>>> import numpy as np
>>> torch.testing.assert_close(
...     [torch.ones(2, 3), np.zeros(3)],
...     [torch.ones(2, 3), np.zeros(3)],
... )
>>> torch.testing.assert_close([torch.ones(2, 3), 42], [torch.ones(2, 3), 42])

```

However, it does not work if the data structure contains a string:

```pycon

>>> import torch
>>> torch.testing.assert_close(
...     {'key1': torch.ones(2, 3), 'key2': torch.zeros(3), "key3": "abc"},
...     {'key1': torch.ones(2, 3), 'key2': torch.zeros(3), "key3": "abc"},
... )
Traceback (most recent call last):
...
TypeError: No comparison pair was able to handle inputs of type <class 'str'> and <class 'str'>.
The failure occurred for item ['key3']

```

`coola` can compare these objects:

```pycon

>>> import torch
>>> import coola
>>> coola.objects_are_equal(
...     {'key1': torch.ones(2, 3), 'key2': torch.zeros(3), "key3": "abc"},
...     {'key1': torch.ones(2, 3), 'key2': torch.zeros(3), "key3": "abc"},
... )
True

```

Internally, [`torch.testing.assert_close`](https://pytorch.org/docs/stable/testing.html) tries to
convert some values to tensors to compare them, which can lead to surprising results like:

```pycon

>>> import torch
>>> torch.testing.assert_close((1, 2, 3), [1, 2, 3])

```

The inputs have different types: the left input is a tuple, whereas the right is a list.
`coola` has a strict type checking and will indicate the two inputs are different:

```pycon

>>> import torch
>>> import coola
>>> coola.objects_are_equal((1, 2, 3), [1, 2, 3])
False

```

[`numpy.testing.assert_equal`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_equal.html)
has different limitations.
For example, it can work with strings but can handle only simple sequence and mapping objects

```pycon

>>> import numpy as np
>>> from collections import deque
>>> np.testing.assert_equal(
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3)},
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3)},
... )
>>> np.testing.assert_equal(
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3), "key3": "abc"},
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3), "key3": "abc"},
... )
>>> np.testing.assert_equal(
...     deque([np.ones((2, 3)), np.zeros(3)]),
...     deque([np.ones((2, 3)), np.zeros(3)]),
... )
Traceback (most recent call last):
...
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

```

`coola` can compare these objects:

```pycon

>>> import coola
>>> import numpy as np
>>> from collections import deque
>>> coola.objects_are_equal(
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3)},
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3)},
... )
True
>>> coola.objects_are_equal(
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3), "key3": "abc"},
...     {'key1': np.ones((2, 3)), 'key2': np.zeros(3), "key3": "abc"},
... )
True
>>> coola.objects_are_equal(
...     deque([np.ones((2, 3)), np.zeros(3)]),
...     deque([np.ones((2, 3)), np.zeros(3)]),
... )
True

```
