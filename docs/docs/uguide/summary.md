# Summarizing Nested Data

:book: This page describes the `coola.summary` package, which provides utilities for creating
human-readable text summaries of complex and nested data structures.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.summary` package provides a type-based system for generating formatted string
representations of Python objects, with special support for nested data structures. Unlike
Python's built-in `repr()` or `str()`, this package offers:

- **Configurable depth control**: Limit how deep nested structures are expanded
- **Type-specific formatting**: Different types are formatted according to their structure
- **Truncation support**: Limit the number of items shown in collections
- **Extensibility**: Register custom summarizers for your own types
- **Automatic type dispatch**: Automatically selects the right summarizer based on type

This is particularly useful for debugging, logging, and displaying complex data in a readable
format without being overwhelmed by deeply nested or large structures.

## Basic Usage

### The `summarize()` Function

The simplest way to create a summary is using the `summarize()` function:

```pycon

>>> from coola.summary import summarize
>>> print(summarize({"a": 1, "b": "abc"}))
<class 'dict'> (length=2)
  (a): 1
  (b): abc

```

The function automatically detects the type of data and formats it appropriately.

### Simple Values

For simple scalar values, the summary shows the type and value:

```pycon

>>> from coola.summary import summarize
>>> print(summarize(42))
<class 'int'> 42
>>> print(summarize(3.14159))
<class 'float'> 3.14159
>>> print(summarize("hello"))
<class 'str'> hello
>>> print(summarize(True))
<class 'bool'> True

```

### Lists and Tuples

Sequences are displayed with their type, length, and indexed items:

```pycon

>>> from coola.summary import summarize
>>> print(summarize([1, 2, 3, 4, 5]))
<class 'list'> (length=5)
  (0): 1
  (1): 2
  (2): 3
  (3): 4
  (4): 5

```

Tuples work the same way:

```pycon

>>> from coola.summary import summarize
>>> print(summarize((10, 20, 30)))
<class 'tuple'> (length=3)
  (0): 10
  (1): 20
  (2): 30

```

### Dictionaries

Dictionaries show their type, length, and key-value pairs:

```pycon

>>> from coola.summary import summarize
>>> print(summarize({"name": "Alice", "age": 30, "city": "NYC"}))
<class 'dict'> (length=3)
  (name): Alice
  (age): 30
  (city): NYC

```

### Sets

Sets display their type, length, and elements:

```pycon

>>> from coola.summary import summarize
>>> print(summarize({1, 2, 3, 4, 5}))
<class 'set'> (length=5)
  (0): 1
  (1): 2
  (2): 3
  (3): 4
  (4): 5

```

Note: Sets are unordered, so the element order in the summary may vary.

## Working with Nested Structures

### Depth Control

The most important feature of `coola.summary` is controlling how deeply nested structures are
expanded using the `max_depth` parameter:

```pycon

>>> from coola.summary import get_default_registry
>>> nested_data = {
...     'level1': {
...         'level2': {
...             'level3': [1, 2, 3]
...         }
...     }
... }
>>> registry = get_default_registry()

```

**max_depth=0**: Shows the raw string representation (no expansion):

```pycon

>>> print(registry.summarize(nested_data, max_depth=0))
{'level1': {'level2': {'level3': [1, 2, 3]}}}

```

**max_depth=1**: Expands only the top level:

```pycon

>>> print(registry.summarize(nested_data, max_depth=1))
<class 'dict'> (length=1)
  (level1): {'level2': {'level3': [1, 2, 3]}}

```

**max_depth=2**: Expands two levels deep:

```pycon

>>> print(registry.summarize(nested_data, max_depth=2))
<class 'dict'> (length=1)
  (level1): <class 'dict'> (length=1)
      (level2): {'level3': [1, 2, 3]}

```

**max_depth=3**: Expands three levels deep:

```pycon

>>> print(registry.summarize(nested_data, max_depth=3))
<class 'dict'> (length=1)
  (level1): <class 'dict'> (length=1)
      (level2): <class 'dict'> (length=1)
          (level3): [1, 2, 3]

```

### Complex Nested Examples

**Nested lists and dictionaries:**

```pycon

>>> from coola.summary import summarize
>>> data = {
...     "users": [
...         {"name": "Alice", "scores": [95, 87, 92]},
...         {"name": "Bob", "scores": [88, 91, 85]}
...     ],
...     "metadata": {"count": 2, "version": "1.0"}
... }
>>> print(summarize(data))
<class 'dict'> (length=2)
  (users): [{'name': 'Alice', 'scores': [95, 87, 92]}, {'name': 'Bob', 'scores': [88, 91, 85]}]
  (metadata): {'count': 2, 'version': '1.0'}

```

**Mixed types:**

```pycon

>>> from coola.summary import summarize
>>> mixed = {
...     "int": 42,
...     "float": 3.14,
...     "string": "hello",
...     "list": [1, 2, 3],
...     "tuple": (4, 5, 6),
...     "set": {7, 8, 9},
...     "dict": {"nested": "value"}
... }
>>> print(summarize(mixed))
<class 'dict'> (length=7)
  (int): 42
  (float): 3.14
  (string): hello
  (list): [1, 2, 3]
  (tuple): (4, 5, 6)
  (set): {8, 9, 7}
  (dict): {'nested': 'value'}

```

## Controlling Output Size

### Limiting Items in Collections

By default, collections show a maximum of 5 items. Longer collections are truncated with `...`:

```pycon

>>> from coola.summary import summarize
>>> long_list = list(range(20))
>>> print(summarize(long_list))
<class 'list'> (length=20)
  (0): 0
  (1): 1
  (2): 2
  (3): 3
  (4): 4
  ...

```

### Custom max_items

You can customize the number of items shown using custom summarizers:

```pycon

>>> from coola.summary import SummarizerRegistry, SequenceSummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(list, SequenceSummarizer(max_items=3))
>>> long_list = list(range(10))
>>> print(registry.summarize(long_list))
<class 'list'> (length=10)
  (0): 0
  (1): 1
  (2): 2
  ...

```

To show all items, set `max_items=-1`:

```pycon

>>> from coola.summary import SummarizerRegistry, SequenceSummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(list, SequenceSummarizer(max_items=-1))
>>> print(registry.summarize(list(range(10))))
<class 'list'> (length=10)
  (0): 0
  (1): 1
  (2): 2
  (3): 3
  (4): 4
  (5): 5
  (6): 6
  (7): 7
  (8): 8
  (9): 9

```

### Truncating Long Strings

For very long strings or values, you can use `DefaultSummarizer` with `max_characters`:

```pycon

>>> from coola.summary import SummarizerRegistry, DefaultSummarizer
>>> long_string = "This is a very long string that should be truncated when max_characters is set"
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer(max_characters=30))
>>> print(registry.summarize(long_string))
<class 'str'> This is a very long string tha...

```

## Customizing Indentation

The `num_spaces` parameter controls indentation for nested structures:

```pycon

>>> from coola.summary import SummarizerRegistry, MappingSummarizer, SequenceSummarizer, DefaultSummarizer
>>> data = {'a': [1, 2, 3], 'b': {'nested': 'value'}}
>>> # Default: 2 spaces
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(list, SequenceSummarizer(num_spaces=2))
>>> registry.register(dict, MappingSummarizer(num_spaces=2))
>>> print(registry.summarize(data, max_depth=3))
<class 'dict'> (length=2)
  (a): <class 'list'> (length=3)
      (0): <class 'int'> 1
      (1): <class 'int'> 2
      (2): <class 'int'> 3
  (b): <class 'dict'> (length=1)
      (nested): <class 'str'> value

```

With 4 spaces for clearer nesting:

```pycon

>>> from coola.summary import SummarizerRegistry, MappingSummarizer, SequenceSummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(list, SequenceSummarizer(num_spaces=4))
>>> registry.register(dict, MappingSummarizer(num_spaces=4))
>>> print(registry.summarize(data, max_depth=3))
<class 'dict'> (length=2)
    (a): <class 'list'> (length=3)
            (0): <class 'int'> 1
            (1): <class 'int'> 2
            (2): <class 'int'> 3
    (b): <class 'dict'> (length=1)
            (nested): <class 'str'> value

```

## Working with NumPy and PyTorch

### NumPy Arrays

The `NDArraySummarizer` creates compact summaries of NumPy arrays:

```pycon

>>> import numpy as np
>>> from coola.summary import SummarizerRegistry, NDArraySummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(np.ndarray, NDArraySummarizer())
>>> arr = np.arange(100).reshape(10, 10)
>>> print(registry.summarize(arr))
<class 'numpy.ndarray'> | shape=(10, 10) | dtype=int64

```

By default, only metadata is shown. To see the actual data, use `show_data=True`:

```pycon

>>> from coola.summary import SummarizerRegistry, NDArraySummarizer, DefaultSummarizer
>>> import numpy as np
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(np.ndarray, NDArraySummarizer(show_data=True))
>>> arr = np.arange(5)
>>> print(registry.summarize(arr))
array([0, 1, 2, 3, 4])

```

### PyTorch Tensors

The `TensorSummarizer` works similarly for PyTorch tensors:

```pycon

>>> import torch
>>> from coola.summary import SummarizerRegistry, TensorSummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(torch.Tensor, TensorSummarizer())
>>> tensor = torch.randn(3, 4, 5)
>>> print(registry.summarize(tensor))
<class 'torch.Tensor'> | shape=torch.Size([3, 4, 5]) | dtype=torch.float32 | device=cpu | requires_grad=False

```

With `show_data=True`:

```pycon

>>> from coola.summary import SummarizerRegistry, TensorSummarizer, DefaultSummarizer
>>> import torch
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(torch.Tensor, TensorSummarizer(show_data=True))
>>> tensor = torch.arange(5)
>>> print(registry.summarize(tensor))
tensor([0, 1, 2, 3, 4])

```

## Creating Custom Summarizers

### Defining a Custom Summarizer

To create a custom summarizer for your own types, extend `BaseSummarizer`:

```pycon

>>> from coola.summary import BaseSummarizer, SummarizerRegistry

>>> class Person:
...     def __init__(self, name, age):
...         self.name = name
...         self.age = age
...
>>> class PersonSummarizer(BaseSummarizer):
...     def equal(self, other: object) -> bool:
...         return type(self) is type(other)
...
...     def summarize(self, data, registry, depth=0, max_depth=1):
...         return f"Person(name={data.name!r}, age={data.age})"
...

```

### Registering Custom Summarizers

Register your custom summarizer with the registry:

```pycon

>>> from coola.summary import register_summarizers
>>> register_summarizers({Person: PersonSummarizer()})
>>> person = Person("Alice", 30)
>>> from coola.summary import summarize
>>> print(summarize(person))
Person(name='Alice', age=30)

```

### Nested Custom Types

Custom summarizers work seamlessly with nested structures:

```pycon

>>> from coola.summary import summarize
>>> team = {
...     "leader": Person("Alice", 30),
...     "members": [Person("Bob", 25), Person("Charlie", 28)]
... }
>>> print(summarize(team))
<class 'dict'> (length=2)
  (leader): Person(name='Alice', age=30)
  (members): [Person(name='Bob', age=25), Person(name='Charlie', age=28)]

```

## Working with the Registry

### Understanding the Registry

The `SummarizerRegistry` manages the mapping from types to summarizers:

```pycon

>>> from coola.summary import get_default_registry
>>> registry = get_default_registry()
>>> registry
SummarizerRegistry(
  (state): TypeRegistry(
      (<class 'object'>): DefaultSummarizer(max_characters=-1)
      (<class 'str'>): DefaultSummarizer(max_characters=-1)
      (<class 'int'>): DefaultSummarizer(max_characters=-1)
      (<class 'float'>): DefaultSummarizer(max_characters=-1)
      (<class 'complex'>): DefaultSummarizer(max_characters=-1)
      (<class 'bool'>): DefaultSummarizer(max_characters=-1)
      (<class 'list'>): SequenceSummarizer(max_items=5, num_spaces=2)
      (<class 'tuple'>): SequenceSummarizer(max_items=5, num_spaces=2)
      (<class 'collections.abc.Sequence'>): SequenceSummarizer(max_items=5, num_spaces=2)
      (<class 'set'>): SetSummarizer(max_items=5, num_spaces=2)
      (<class 'frozenset'>): SetSummarizer(max_items=5, num_spaces=2)
      (<class 'dict'>): MappingSummarizer(max_items=5, num_spaces=2)
      (<class 'collections.abc.Mapping'>): MappingSummarizer(max_items=5, num_spaces=2)
    )
)

```

### Creating a Custom Registry

For complete control, create your own registry:

```pycon

>>> from coola.summary import SummarizerRegistry, SequenceSummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(list, SequenceSummarizer())
>>> print(registry.summarize([1, 2, 3]))
<class 'list'> (length=3)
  (0): 1
  (1): 2
  (2): 3

```

### Registering Multiple Types

Use `register_many()` to register multiple types at once:

```pycon

>>> from coola.summary import SummarizerRegistry, SequenceSummarizer, MappingSummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register_many({
...     object: DefaultSummarizer(),
...     list: SequenceSummarizer(),
...     dict: MappingSummarizer()
... })

```

### Checking Registered Types

Check if a type has a registered summarizer:

```pycon

>>> from coola.summary import get_default_registry
>>> registry = get_default_registry()
>>> registry.has_summarizer(list)
True
>>> registry.has_summarizer(int)
True

```

## Common Use Cases

### Debugging Complex Data

Quickly inspect complex data structures during debugging:

```pycon

>>> from coola.summary import summarize
>>> config = {
...     "database": {
...         "host": "localhost",
...         "port": 5432,
...         "credentials": {"user": "admin", "password": "secret"}
...     },
...     "cache": {
...         "enabled": True,
...         "ttl": 3600,
...         "backends": ["redis", "memcached"]
...     }
... }
>>> print(summarize(config))
<class 'dict'> (length=2)
  (database): {'host': 'localhost', 'port': 5432, 'credentials': {'user': 'admin', 'password': 'secret'}}
  (cache): {'enabled': True, 'ttl': 3600, 'backends': ['redis', 'memcached']}

```

### Logging Large Tensors

Log tensor shapes without filling logs with data:

```pycon

>>> import torch
>>> from coola.summary import SummarizerRegistry, TensorSummarizer, DefaultSummarizer
>>> registry = SummarizerRegistry()
>>> registry.register(object, DefaultSummarizer())
>>> registry.register(torch.Tensor, TensorSummarizer())
>>> model_weights = {
...     "layer1": torch.randn(1000, 1000),
...     "layer2": torch.randn(1000, 500),
...     "bias": torch.randn(500)
... }
>>> print(registry.summarize(model_weights))
<class 'dict'> (length=3)
  (layer1): <class 'torch.Tensor'> | shape=torch.Size([1000, 1000]) | dtype=torch.float32 | device=cpu | requires_grad=False
  (layer2): <class 'torch.Tensor'> | shape=torch.Size([1000, 500]) | dtype=torch.float32 | device=cpu | requires_grad=False
  (bias): <class 'torch.Tensor'> | shape=torch.Size([500]) | dtype=torch.float32 | device=cpu | requires_grad=False

```

### Inspecting API Responses

Summarize complex API responses:

```pycon

>>> from coola.summary import summarize
>>> api_response = {
...     "status": "success",
...     "data": {
...         "items": [
...             {"id": 1, "name": "Item 1", "tags": ["tag1", "tag2"]},
...             {"id": 2, "name": "Item 2", "tags": ["tag3"]},
...             # ... potentially many more items
...         ],
...         "pagination": {"page": 1, "total_pages": 10}
...     }
... }
>>> print(summarize(api_response))
<class 'dict'> (length=2)
  (status): success
  (data): {'items': [{'id': 1, 'name': 'Item 1', 'tags': ['tag1', 'tag2']}, {'id': 2, 'name': 'Item 2', 'tags': ['tag3']}], 'pagination': {'page': 1, 'total_pages': 10}}

```

### Comparing Data Structures

Get a quick overview to compare different data structures. Note that with default `max_depth=1`,
nested structures are shown as raw strings without truncation:

```pycon

>>> from coola.summary import summarize
>>> data1 = {"users": [1, 2, 3, 4, 5], "version": "1.0"}
>>> data2 = {"users": [1, 2, 3], "version": "2.0"}
>>> print("Data 1:")
>>> print(summarize(data1))
Data 1:
<class 'dict'> (length=2)
  (users): [1, 2, 3, 4, 5]
  (version): 1.0
>>> print("Data 2:")
>>> print(summarize(data2))
Data 2:
<class 'dict'> (length=2)
  (users): [1, 2, 3]
  (version): 2.0

```

For deeper inspection with truncation, increase `max_depth`:

```pycon

>>> from coola.summary import get_default_registry
>>> registry = get_default_registry()
>>> data = {"users": list(range(20)), "version": "1.0"}
>>> print(registry.summarize(data, max_depth=2))
<class 'dict'> (length=2)
  (users): <class 'list'> (length=20)
      (0): 0
      (1): 1
      (2): 2
      (3): 3
      (4): 4
      ...
  (version): 1.0

```

## Available Summarizers

The `coola.summary` package provides the following built-in summarizers:

- **`DefaultSummarizer`**: For generic objects and scalar types (int, float, str, bool, etc.)
  - Configurable: `max_characters` (default: -1, no limit)
- **`SequenceSummarizer`**: For sequences (list, tuple, Sequence ABC)
  - Configurable: `max_items` (default: 5), `num_spaces` (default: 2)
- **`MappingSummarizer`**: For mappings (dict, Mapping ABC)
  - Configurable: `max_items` (default: 5), `num_spaces` (default: 2)
- **`SetSummarizer`**: For sets (set, frozenset)
  - Configurable: `max_items` (default: 5), `num_spaces` (default: 2)
- **`NDArraySummarizer`**: For NumPy arrays (requires NumPy)
  - Configurable: `show_data` (default: False)
- **`TensorSummarizer`**: For PyTorch tensors (requires PyTorch)
  - Configurable: `show_data` (default: False)

## Design Principles

The `coola.summary` package is designed with the following principles:

1. **Type-based dispatch**: Automatically selects the appropriate summarizer based on data type
2. **Recursive summarization**: Handles deeply nested structures through the registry pattern
3. **Configurable output**: Control depth, item limits, and formatting to suit your needs
4. **Extensibility**: Easy to add support for custom types via the registry
5. **Sensible defaults**: Works out-of-the-box for common Python types
6. **Metadata focus**: For large data structures (arrays, tensors), show metadata instead of data

## See Also

- [Recursive Data Transformation](recursive.md): For transforming nested data while preserving structure
- [Iterating Over Nested Data](iterator.md): For iterating over nested data structures
- [Registry System](registry.md): For understanding the registry pattern used internally
