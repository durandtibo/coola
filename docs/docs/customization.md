# Customization

Both `objects_are_equal` and `objects_are_allclose` functions can be easily customized.

## How to implement a custom `BaseEqualityTester`

Internally, the `objects_are_equal` function uses a `BaseEqualityTester` object to check if two
objects are equal.
`coola` comes with a default `BaseEqualityTester` named `EqualityTester`, but it is possible to
implement a custom `BaseEqualityTester` to check if two objects are equal.
The following example shows how to use a custom `BaseEqualityTester`.

```pycon
>>> from typing import Any
>>> from coola import objects_are_equal
>>> from coola.equality import EqualityConfig
>>> from coola.equality.testers import BaseEqualityTester
>>> class MyCustomEqualityTester(BaseEqualityTester):
...     def equal(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
...         return object1 is object2
...
>>> objects_are_equal([1, 2, 3], (1, 2, 3), tester=MyCustomEqualityTester())
False

```

Implementing a new `BaseEqualityTester` allows to customize the behavior of `objects_are_equal`.

## How to customize `EqualityTester`

Implementing a new `BaseEqualityTester` can be a lot of work, so it is not always a practical
solution.
For example if you want to support a new type, you do not want to reimplement everything.
Instead of implementing a new `BaseEqualityTester`, it is possible to customize the
default `EqualityTester`.

### Overview

`EqualityTester` has a registry of equality comparators with their associated types.
An equality comparator is an object that follows the `BaseEqualityComparator` interface.
`EqualityTester` uses the Method Resolution Order (MRO) of the first object to find the equality
comparator to use.
It uses the most specific equality comparator.
For example, `EqualityTester` has an equality comparator registered for `object` and another
one `list`.
If the first element to compare is a `list`, `EqualityTester` will use the equality comparator
associated to `list` to compare the two objects.
You can use the following code to see the registered equality comparators with their associated types.

```pycon
>>> from coola.equality.testers import EqualityTester
>>> EqualityTester.registry
{<class 'object'>: DefaultEqualityComparator(),
 <class 'collections.abc.Mapping'>: MappingEqualityComparator(),
 <class 'collections.abc.Sequence'>: SequenceEqualityComparator(),
 <class 'collections.deque'>: SequenceEqualityComparator(),
 <class 'dict'>: MappingEqualityComparator(),
 <class 'list'>: SequenceEqualityComparator(),
 <class 'tuple'>: SequenceEqualityComparator(),
 ...}

```

An equality comparator (`DefaultEqualityOperator`) is registered for `object` type, so this equality
comparator is considered like the default equality comparator.
For example, it will be used to compare `int` or `float` or `str` because there is no specific
equality comparator for these types.
Note that the same equality comparator can be used for multiple types.
For example, by default, the same equality comparator is used for `list`, `tuple`,
and `collections.abc.Sequence`.
The following sections explain how to customize this registry.

### Add an equality comparator

It is possible to add a new equality comparator to the `EqualityTester`.
The following example shows how to define a new behavior for strings.
Instead of checking if two strings are the same (default behavior), the new behavior is that two
strings are equal if the first string is a part of the second string.
It is a two-steps process to add a new equality comparator to `EqualityTester`.
First, you need to implement a new `BaseEqualityComparator` with the expected behavior for the
specific type (`str` for this example).
Then, you need to add the `BaseEqualityComparator` to `EqualityTester`.

```pycon
>>> from typing import Any
>>> from coola import objects_are_equal
>>> from coola.equality.comparators import BaseEqualityComparator
>>> from coola.equality.testers import BaseEqualityTester, EqualityTester
>>> # Step 1: implementation of a new equality comparator
>>> class MyCustomStrEqualityOperator(BaseEqualityComparator):
...     def clone(self) -> "MyCustomStrEqualityOperator":
...         return self.__class__()
...
...     def equal(self, object1: str, object2: Any, config: EqualityConfig) -> bool:
...         # You can add code to check the type and to log a message to indicate
...         # the difference between the objects if any. To keep this example
...         # simple, this part is skipped.
...         return object1 in object2
...
>>> # Step 2: add the new equality comparator to EqualityTester
>>> tester = EqualityTester.local_copy()
>>> tester.add_comparator(str, MyCustomStrEqualityOperator())
>>> objects_are_equal("abc", "abcde", tester=tester)
True
>>> objects_are_equal("abc", "cba", tester=tester)
False
>>> tester.registry[str]
MyCustomStrEqualityOperator()

```

Once registered, the new equality comparator is used automatically when you use
the `objects_are_equal` function.
You can use the `registry` attribute to check the registered equality comparators.
You should see the new added equality comparator (last line for this example).

### Update the equality comparator for a given type

The previous section explains how to add a new equality comparator to `EqualityTester`.
This section explains how to update the equality comparator for a specific type.
To update an equality comparator for a given type, you need to add the argument `exist_ok=True` when
the new equality comparator is added.

```pycon
>>> from collections.abc import Mapping
>>> from coola.equality.comparators import BaseEqualityComparator
>>> from coola.equality.testers import EqualityTester
>>> class MyCustomMappingEqualityComparator(BaseEqualityComparator):
...     def clone(self) -> "MyCustomMappingEqualityComparator":
...         return self.__class__()
...
...     def equal(self, object1: Mapping, object2: Any, config: EqualityConfig) -> bool:
...         # You can add code to check the type and to log a message to indicate
...         # the difference between the objects if any. To keep this example
...         # simple, this part is skipped.
...         return object1 is object2
...
>>> tester = EqualityTester.local_copy()
>>> tester.add_comparator(
...     Mapping,
...     MyCustomMappingEqualityComparator(),
...     exist_ok=True,
... )
>>> tester.registry[Mapping]
MyCustomMappingEqualityComparator()

```
