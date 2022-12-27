# coola quickstart

:book: This page is a quick overview of the two main functions of `coola`: `objects_are_equal`
and `objects_are_allclose`. The motivation of the library is explained [here](index.md#motivation).
You should read this page if you want to learn how to use these functions. This page does not
explain the internal behavior of these functions.

**Prerequisites:** You’ll need to know a bit of Python. For a refresher, see
the [Python tutorial](https://docs.python.org/tutorial/).
It is highly recommended to know a bit of [NumPy](https://numpy.org/doc/stable/user/quickstart.html)
or [PyTorch](https://pytorch.org/tutorials/).

## Equal or not?

`coola` provides a function `objects_are_equal` that can indicate if two complex/nested objects are
equal or not. It also works for simple objects like integer or string. The following example shows
how to use this function.

```python
import numpy
import torch

from coola import objects_are_equal

data1 = {'torch': torch.ones(2, 3), 'numpy': numpy.zeros((2, 3))}
data2 = {'torch': torch.zeros(2, 3), 'numpy': numpy.ones((2, 3))}
data3 = {'torch': torch.ones(2, 3), 'numpy': numpy.zeros((2, 3))}

print(objects_are_equal(data1, data2))
print(objects_are_equal(data1, data3))
```

*Output*:

```textmate
False
True
```

When the objects are complex or nested, it is not obvious to know which element is different.
This function has an argument `show_difference` which shows the first difference found between the
two objects. For example if you add `show_difference=True` when you compare the `data1`
and `data2`, you will see at least one element that is different:

```python
objects_are_equal(data1, data2, show_difference=True)
```

*Output*:

```textmate
INFO:coola.pytorch:torch.Tensors are different
object1=
tensor([[1., 1., 1.],
        [1., 1., 1.]])
object2=
tensor([[0., 0., 0.],
        [0., 0., 0.]])
INFO:coola.equality:The mappings have a different value for the key 'torch':
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

The previous examples use dictionary, but it is possible to use other types like list or tuple

```python
data1 = [torch.ones(2, 3), numpy.zeros((2, 3))]
data2 = [torch.zeros(2, 3), numpy.ones((2, 3))]
data3 = (torch.ones(2, 3), numpy.zeros((2, 3)))

print(objects_are_equal(data1, data2))
print(objects_are_equal(data1, data3))
```

*Output*:

```textmate
False
False
```

It is also possible to test more complex objects

```python
data1 = {'list': [torch.ones(2, 3), numpy.zeros((2, 3))],
         'dict': {'torch': torch.arange(5), 'str': 'abc'}, 'int': 1}
data2 = {'list': [torch.ones(2, 3), numpy.zeros((2, 3))],
         'dict': {'torch': torch.arange(5), 'str': 'abcd'}, 'int': 1}

print(objects_are_equal(data1, data2))
```

*Output*:

```textmate
False
```

Feel free to try any complex nested structure that you want. You can find the currently supported
types [here](types.md).
