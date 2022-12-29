# Customization

Both `objects_are_equal` and `objects_are_allclose` functions can be easily customized.

## Mechanism/Overview

Internally, the `objects_are_equal` function uses a `BaseEqualityTester` to check if two
objects are equal.
`coola` comes with a default `BaseEqualityTester` named `EqualityTester`, but it is possible to
implement a custom `BaseEqualityTester` to check if two objects are equal.
The following example shows how to use a custom `BaseEqualityTester`.

```python
from coola import BaseEqualityTester, objects_are_equal


class MyCustomEqualityTester(BaseEqualityTester):
    ...  # Custom implementation


objects_are_equal([1, 2, 3], (1, 2, 3), tester=MyCustomEqualityTester())
```

## How to customize `EqualityTester`

Implementing a new `BaseEqualityTester` can be a lot of work so it is not always a practical
solution.
Instead of implementing a new `BaseEqualityTester`, it is possible to customize the
default `EqualityTester`.

### Overview

TODO

### Add an equality operator

It is possible to add a new equality operator to the `EqualityTester`.
The following example shows how to define a new behavior for strings.
Instead of checking if two strings are the same (default behavior), the new behavior is that two
strings are equal if the first string is a part of the second string.
It is a two-steps process to add a new equality operator to `EqualityTester`.
First, you need to implement a new `BaseEqualityOperator` with the expected behavior for the
specific type (`str` for this example).
Then, you need to add the `BaseEqualityOperator` to `EqualityTester`.

```python
from typing import Any

from coola import BaseEqualityOperator, BaseEqualityTester, EqualityTester, objects_are_equal


# Step 1: implementation of a new equality operator
class MyCustomStrEqualityOperator(BaseEqualityOperator):

    def equal(
            self,
            tester: BaseEqualityTester,
            object1: str,
            object2: Any,
            show_difference: bool = False,
    ) -> bool:
        # You can add code to check the type and to log a message to indicate 
        # the difference between the objects if any. To keep this example 
        # simple, this part is skipped. 
        return object1 in object2


# Step 2: add the new equality operator to EqualityTester
EqualityTester.add_equality_operator(str, MyCustomStrEqualityOperator())

print(objects_are_equal('abc', 'abcde'))
print(objects_are_equal('abc', 'cba'))
```

*Output*:

```textmate
True
False
```

Once registered, the new equality operator is used automatically when you use
the `objects_are_equal` function.
You can use the `registry` attribute to check the registered equality operators.

```python
EqualityTester.registry
```

*Output*:

```textmate
{collections.abc.Mapping: MappingEqualityOperator(),
 collections.abc.Sequence: SequenceEqualityOperator(),
 dict: MappingEqualityOperator(),
 list: SequenceEqualityOperator(),
 object: DefaultEqualityOperator(),
 tuple: SequenceEqualityOperator(),
 numpy.ndarray: NDArrayEqualityOperator(),
 torch.nn.utils.rnn.PackedSequence: PackedSequenceEqualityOperator(),
 torch.Tensor: TensorEqualityOperator(),
 str: MyCustomStrEqualityOperator()}
```

You should see the new added equality operator (last line for this example).

### Update the equality operator for a given type

The previous section explains how to add a new equality operator to `EqualityTester`.
This section explains how to update the equality operator for a specific type.
To update an equality operator for a given type, you need to add the argument `exist_ok=True` when
the new equality operator is added.

```python
from collections.abc import Mapping

from coola import BaseEqualityOperator, EqualityTester


class MyCustomMappingEqualityOperator(BaseEqualityOperator):
    ...  # Custom implementation


EqualityTester.add_equality_operator(
    Mapping,
    MyCustomMappingEqualityOperator(),
    exist_ok=True,
)
```

To see the difference, you can check the list of registered equality operators before and after.
Before, the registered equality operators should look like:

```textmate
{collections.abc.Mapping: MappingEqualityOperator(),
 collections.abc.Sequence: SequenceEqualityOperator(),
 dict: MappingEqualityOperator(),
 list: SequenceEqualityOperator(),
 object: DefaultEqualityOperator(),
 tuple: SequenceEqualityOperator(),
 numpy.ndarray: NDArrayEqualityOperator(),
 torch.nn.utils.rnn.PackedSequence: PackedSequenceEqualityOperator(),
 torch.Tensor: TensorEqualityOperator()}
```

After, the registered equality operators should look like:

```textmate
{collections.abc.Mapping: MyCustomMappingEqualityOperator(),
 collections.abc.Sequence: SequenceEqualityOperator(),
 dict: MappingEqualityOperator(),
 list: SequenceEqualityOperator(),
 object: DefaultEqualityOperator(),
 tuple: SequenceEqualityOperator(),
 numpy.ndarray: NDArrayEqualityOperator(),
 torch.nn.utils.rnn.PackedSequence: PackedSequenceEqualityOperator(),
 torch.Tensor: TensorEqualityOperator()}
```