# Working with Nested Data Structures

:book: This page describes the `coola.nested` package, which provides utilities for manipulating
nested data structures like converting between different formats and working with nested mappings.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.nested` package provides utilities for working with nested data structures. It offers:

1. **Data conversion functions** - Convert between list-of-dicts and dict-of-lists formats
2. **Mapping utilities** - Flatten nested dictionaries, extract values, filter keys, rename keys,
   merge mappings and flatten a mapping of mappings
3. **JSON conversion** - Convert a nested structure to JSON-compatible data
4. **Clean and simple API** - Easy-to-use functions for common data transformation tasks

This is particularly useful when working with tabular data, configuration files, or any nested data
structures that need to be transformed or manipulated.

## Data Conversion

### Converting Between Formats

The package provides two main conversion functions for transforming between different
representations of tabular data:

#### List of Dicts to Dict of Lists

Use `convert_to_dict_of_lists()` to convert a sequence of dictionaries (rows) into a dictionary of
lists (columns):

```pycon
>>> from coola.nested import convert_to_dict_of_lists
>>> rows = [
...     {"name": "Alice", "age": 30, "city": "NYC"},
...     {"name": "Bob", "age": 25, "city": "LA"},
...     {"name": "Charlie", "age": 35, "city": "Chicago"},
... ]
>>> convert_to_dict_of_lists(rows)
{'name': ['Alice', 'Bob', 'Charlie'], 'age': [30, 25, 35], 'city': ['NYC', 'LA', 'Chicago']}

```

This is useful when you need to work with columnar data formats or when preparing data for certain
libraries that expect column-oriented data.

**Important:** All dictionaries in the sequence should have the same keys. The function uses the
first dictionary's keys to determine which keys to extract.

#### Dict of Lists to List of Dicts

Use `convert_to_list_of_dicts()` to convert a dictionary of sequences (columns) into a list of
dictionaries (rows):

```pycon
>>> from coola.nested import convert_to_list_of_dicts
>>> columns = {
...     "name": ["Alice", "Bob", "Charlie"],
...     "age": [30, 25, 35],
...     "city": ["NYC", "LA", "Chicago"],
... }
>>> convert_to_list_of_dicts(columns)
[{'name': 'Alice', 'age': 30, 'city': 'NYC'}, {'name': 'Bob', 'age': 25, 'city': 'LA'}, {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}]

```

This is useful when you have columnar data and need to process it row-by-row, or when interfacing
with APIs that expect row-oriented data.

**Important:** All sequences in the dictionary should have the same length.

### Round-Trip Conversion

These conversion functions are inverse operations of each other:

```pycon
>>> from coola.nested import convert_to_dict_of_lists, convert_to_list_of_dicts
>>> original = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
>>> columns = convert_to_dict_of_lists(original)
>>> columns
{'x': [1, 3], 'y': [2, 4]}
>>> rows = convert_to_list_of_dicts(columns)
>>> rows
[{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]

```

## Mapping Utilities

### Getting the First Value

Use `get_first_value()` to extract the first value from a mapping:

```pycon
>>> from coola.nested import get_first_value
>>> data = {"first": 10, "second": 20, "third": 30}
>>> get_first_value(data)
10

```

This is useful when you need to extract a representative value from a dictionary, or when you know
the mapping should have at least one value but don't know the key.

**Note:** If the mapping is empty, this function raises a `ValueError`.

```pycon
>>> from coola.nested import get_first_value
>>> get_first_value({})
Traceback (most recent call last):
    ...
ValueError: First value cannot be returned because the mapping is empty

```

### Flattening Nested Dictionaries

Use `to_flat_dict()` to convert nested dictionaries into a flat dictionary with dot-separated keys:

```pycon
>>> from coola.nested import to_flat_dict
>>> nested = {
...     "database": {
...         "host": "localhost",
...         "port": 5432,
...         "credentials": {
...             "username": "admin",
...             "password": "secret",
...         },
...     },
...     "debug": True,
... }
>>> to_flat_dict(nested)
{'database.host': 'localhost', 'database.port': 5432, 'database.credentials.username': 'admin', 'database.credentials.password': 'secret', 'debug': True}

```

This is particularly useful for:

- Configuration management
- Logging hierarchical data
- Creating command-line arguments from nested configurations
- Storing nested data in flat storage systems

#### Customizing the Separator

You can change the separator used to join keys:

```pycon
>>> from coola.nested import to_flat_dict
>>> nested = {"a": {"b": {"c": 1}}}
>>> to_flat_dict(nested, separator="/")
{'a/b/c': 1}

```

#### Handling Lists and Tuples

The function also handles lists and tuples by using numeric indices:

```pycon
>>> from coola.nested import to_flat_dict
>>> data = {
...     "items": [10, 20, 30],
...     "matrix": [[1, 2], [3, 4]],
... }
>>> to_flat_dict(data)
{'items.0': 10, 'items.1': 20, 'items.2': 30, 'matrix.0.0': 1, 'matrix.0.1': 2, 'matrix.1.0': 3, 'matrix.1.1': 4}

```

#### Mixed Nested Structures

The function works with any combination of dicts, lists, and tuples:

```pycon
>>> from coola.nested import to_flat_dict
>>> complex_data = {
...     "users": [
...         {"name": "Alice", "scores": [95, 87]},
...         {"name": "Bob", "scores": [78, 92]},
...     ],
... }
>>> to_flat_dict(complex_data)
{'users.0.name': 'Alice', 'users.0.scores.0': 95, 'users.0.scores.1': 87, 'users.1.name': 'Bob', 'users.1.scores.0': 78, 'users.1.scores.1': 92}

```

`str`, `bytes` and `bytearray` values are leaves: they are never split into characters or bytes.

#### Restoring Lists

`from_flat_dict()` keeps the integer segments as dictionary keys by default, so a list does not come
back as a list. Use `restore_lists=True` to rebuild the lists. A dictionary whose keys are exactly
`"0"`, `"1"`, ..., `"n-1"` is converted, and the tuples come back as lists:

```pycon
>>> from coola.nested import from_flat_dict
>>> flat = {"items.0": 10, "items.1": 20, "matrix.0.0": 1, "matrix.1.0": 3}
>>> from_flat_dict(flat)
{'items': {'0': 10, '1': 20}, 'matrix': {'0': {'0': 1}, '1': {'0': 3}}}
>>> from_flat_dict(flat, restore_lists=True)
{'items': [10, 20], 'matrix': [[1], [3]]}

```

#### Converting Types to Strings

Use the `to_str` parameter to specify types that should be converted to strings instead of being
flattened:

```pycon
>>> from coola.nested import to_flat_dict
>>> data = {
...     "items": [1, 2, 3],
...     "config": {"values": [4, 5, 6]},
... }
>>> to_flat_dict(data, to_str=(list,))
{'items': '[1, 2, 3]', 'config.values': '[4, 5, 6]'}

```

This is useful when you want to preserve certain data types as strings in the flattened output.

### Filtering Keys by Prefix

Use `remove_keys_starting_with()` to filter out dictionary keys that start with a specific prefix:

```pycon
>>> from coola.nested import remove_keys_starting_with
>>> data = {
...     "temp_var1": 100,
...     "temp_var2": 200,
...     "result": 42,
...     "temp_cache": [1, 2, 3],
...     "final_output": "done",
... }
>>> remove_keys_starting_with(data, "temp_")
{'result': 42, 'final_output': 'done'}

```

This is useful for:

- Removing temporary or internal variables
- Filtering configuration keys
- Cleaning up data structures

**Note:** Only string keys are checked for the prefix. Non-string keys are preserved:

```pycon
>>> from coola.nested import remove_keys_starting_with
>>> mixed_keys = {
...     "prefix_str": 1,
...     123: 2,
...     ("a", "b"): 3,
... }
>>> remove_keys_starting_with(mixed_keys, "prefix_")
{123: 2, ('a', 'b'): 3}

```

### Filtering Keys by Condition

Use `remove_keys_containing()` to recursively remove the keys that contain a substring, and
`remove_keys_if()` to recursively remove the keys that satisfy any predicate:

```pycon
>>> from coola.nested import remove_keys_containing, remove_keys_if
>>> remove_keys_containing({"a_tmp": 1, "b": {"c_tmp": 2, "d": 3}}, "tmp")
{'b': {'d': 3}}
>>> remove_keys_if({"a": 1, 2: 2, "b": {"c": 3}}, lambda key: isinstance(key, int))
{'a': 1, 'b': {'c': 3}}

```

Like `remove_keys_starting_with()`, they return a new mapping and keep the input unchanged.

### Adding a Prefix or a Suffix to the Keys

Use `add_prefix_suffix_to_keys()` to rename the keys of a mapping. By default only the top-level keys
are renamed. Use `recursive=True` to rename the keys of the nested mappings too:

```pycon
>>> from coola.nested import add_prefix_suffix_to_keys
>>> data = {"a": 1, "b": {"c": 2}}
>>> add_prefix_suffix_to_keys(data, prefix="x_")
{'x_a': 1, 'x_b': {'c': 2}}
>>> add_prefix_suffix_to_keys(data, prefix="x_", suffix="_y", recursive=True)
{'x_a_y': 1, 'x_b_y': {'x_c_y': 2}}

```

### Merging Mappings

Use `merge_mappings()` to merge an iterable of mappings into a single dictionary. The
`on_duplicate` parameter controls what happens when a key appears in several mappings:

| `on_duplicate`      | Behavior                                                                      |
| ------------------- | ----------------------------------------------------------------------------- |
| `"raise"` (default) | Raise a `DuplicateKeyError`, unless the values are equal.                     |
| `"first"`           | Keep the value of the first mapping.                                          |
| `"last"`            | Keep the value of the last mapping.                                           |
| `"suffix"`          | Keep all the values, adding a numeric suffix to the keys (`a`, `a_1`, `a_2`). |

```pycon
>>> from coola.nested import merge_mappings
>>> merge_mappings([{"a": 1}, {"b": 2}])
{'a': 1, 'b': 2}
>>> merge_mappings([{"a": 1}, {"a": 2}], on_duplicate="last")
{'a': 2}
>>> merge_mappings([{"a": 1}, {"a": 2}], on_duplicate="suffix")
{'a': 1, 'a_1': 2}
>>> merge_mappings([{"a": 1}, {"a": 2}])
Traceback (most recent call last):
    ...
coola.nested.mapping.DuplicateKeyError: Duplicate key found: 'a'

```

Use `equal_nan=True` to consider two `NaN` values equal when the duplicate keys are compared.

### Flattening a Mapping of Mappings

Use `flatten_mapping()` to flatten a mapping whose values are mappings (for example, the metrics of
several splits) into a single dictionary. `on_duplicate` takes `"raise"` (default), `"first"`,
`"last"` or `"prefix"`. With `"prefix"`, all the values are kept and the duplicated keys are renamed
`<outer key><separator><inner key>`. Use `always_prefix=True` to rename every key:

```pycon
>>> from coola.nested import flatten_mapping
>>> metrics = {"train": {"loss": 1, "acc": 2}, "val": {"loss": 3}}
>>> flatten_mapping(metrics, on_duplicate="prefix")
{'acc': 2, 'train.loss': 1, 'val.loss': 3}
>>> flatten_mapping(metrics, always_prefix=True, separator="/")
{'train/loss': 1, 'train/acc': 2, 'val/loss': 3}
>>> flatten_mapping(metrics, on_duplicate="last")
{'loss': 3, 'acc': 2}

```

With the default `on_duplicate="raise"`, a `DuplicateKeyError` is raised for the key `loss`.

## JSON Conversion

Use `convert_to_jsonable()` to recursively convert a nested structure to JSON-compatible data.
Dataclasses and pydantic models found at any depth are converted to dictionaries, and sets are
converted to sorted lists:

```pycon
>>> from dataclasses import dataclass
>>> from coola.nested import convert_to_jsonable
>>> @dataclass
... class Point:
...     x: int
...     y: int
...
>>> convert_to_jsonable([Point(x=1, y=2), {"key": Point(x=3, y=4)}, {3, 1, 2}])
[{'x': 1, 'y': 2}, {'key': {'x': 3, 'y': 4}}, [1, 2, 3]]

```

`try_sorted_list()` converts an iterable (e.g. a set) to a list. The list is sorted when the values
are comparable, so the output is deterministic, otherwise it keeps the iteration order:

```pycon
>>> from coola.nested import try_sorted_list
>>> try_sorted_list({3, 1, 2})
[1, 2, 3]

```

## Common Use Cases

### Working with API Responses

Convert API responses from different formats:

```pycon
>>> from coola.nested import convert_to_dict_of_lists
>>> # API returns list of user objects
>>> api_response = [
...     {"id": 1, "name": "Alice", "active": True},
...     {"id": 2, "name": "Bob", "active": False},
... ]
>>> # Convert to columnar format for analysis
>>> columns = convert_to_dict_of_lists(api_response)
>>> columns
{'id': [1, 2], 'name': ['Alice', 'Bob'], 'active': [True, False]}

```

### Configuration File Processing

Flatten nested configuration for easier access:

```pycon
>>> from coola.nested import to_flat_dict
>>> config = {
...     "server": {
...         "host": "0.0.0.0",
...         "port": 8080,
...         "ssl": {"enabled": True, "cert": "/path/to/cert"},
...     },
...     "logging": {
...         "level": "INFO",
...         "file": "/var/log/app.log",
...     },
... }
>>> flat_config = to_flat_dict(config)
>>> # Easy to access deeply nested values
>>> flat_config["server.ssl.enabled"]
True
>>> flat_config["logging.level"]
'INFO'

```

### Data Cleaning

Remove temporary or internal keys from data:

```pycon
>>> from coola.nested import remove_keys_starting_with
>>> processed_data = {
...     "result": [1, 2, 3],
...     "_internal_state": {"x": 1},
...     "_cache": [4, 5],
...     "output": "success",
... }
>>> clean_data = remove_keys_starting_with(processed_data, "_")
>>> clean_data
{'result': [1, 2, 3], 'output': 'success'}

```

### Batch Processing

Process rows of data individually:

```pycon
>>> from coola.nested import convert_to_list_of_dicts
>>> # Columnar data from database or CSV
>>> columns = {
...     "id": [1, 2, 3],
...     "value": [10, 20, 30],
... }
>>> # Process each row
>>> for row in convert_to_list_of_dicts(columns):
...     print(f"Processing ID {row['id']} with value {row['value']}")
Processing ID 1 with value 10
Processing ID 2 with value 20
Processing ID 3 with value 30

```

## Design Principles

The `coola.nested` package follows these design principles:

1. **Simple and focused**: Each function does one thing well
2. **Type-agnostic**: Works with generic Python types (dict, list, tuple)
3. **Non-destructive**: All functions return new data structures; original data is unchanged
4. **Predictable**: Consistent behavior across different data structures

## Properties

| Property         | Statement                                                                                                |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| Non-destructive  | The functions return new data structures. The input is not modified.                                     |
| Type-agnostic    | They work with the generic Python types (`dict`, `list`, `tuple`), not with a specific library.          |
| Predictable      | The same operation behaves the same way on every supported structure.                                    |
| Explicit failure | An operation without a defined result raises, e.g. `ValueError` for the first value of an empty mapping. |

## See Also

- [`coola.recursive`](recursive.md): For recursively transforming nested data
- [`coola.iterator`](iterator.md): For iterating over nested data structures
- [`coola.equality`](equality.md): For comparing nested data structures
