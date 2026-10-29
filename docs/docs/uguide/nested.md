# Working with Nested Data Structures

:book: This page describes the `coola.nested` package, which provides utilities for manipulating
nested data structures like converting between different formats and working with nested mappings.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.nested` package provides utilities for working with nested data structures. It offers:

1. **Data conversion functions** - Convert between list-of-dicts and dict-of-lists formats
2. **Flat dictionaries** - Flatten a nested structure to dot-separated keys and rebuild it
3. **Mapping utilities** - Extract values, filter keys, rename keys, merge mappings and flatten a
   mapping of mappings
4. **JSON conversion** - Convert a nested structure to JSON-compatible data
5. **Frame flattening** - Flatten the struct and list columns of a polars `DataFrame`

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

**Important:** All sequences in the dictionary should have the same length. A value must be a `Sequence`: a `str`, a set
(its elements have no position) or a generator raises a `TypeError` that names the type.
`convert_to_dict_of_lists()` accepts any iterable of mappings, a generator included.

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

Use `get_first_value()` of `coola.utils.mapping` to extract the first value from a mapping:

```pycon
>>> from coola.utils.mapping import get_first_value
>>> data = {"first": 10, "second": 20, "third": 30}
>>> get_first_value(data)
10

```

This is useful when you need to extract a representative value from a dictionary, or when you know
the mapping should have at least one value but don't know the key.

**Note:** If the mapping is empty, this function raises a `ValueError`.

```pycon
>>> from coola.utils.mapping import get_first_value
>>> get_first_value({})
Traceback (most recent call last):
    ...
ValueError: First value cannot be returned because the mapping is empty

```

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

With `"suffix"`, a generated key never takes the name of a key of an input mapping, whatever the
position of its mapping, so the result does not depend on the order of the mappings:

```pycon
>>> from coola.nested import merge_mappings
>>> merge_mappings([{"a": 1}, {"a": 2}, {"a_1": 3}], on_duplicate="suffix")
{'a': 1, 'a_2': 2, 'a_1': 3}

```

### Flattening a Mapping of Mappings

Use `flatten_mapping()` to flatten a mapping whose values are mappings (for example, the metrics of
several splits) into a single dictionary. `on_duplicate` takes `"raise"` (default), `"first"`,
`"last"` or `"prefix"`. With `"prefix"`, all the values are kept and the duplicated keys are renamed
`<outer key><separator><inner key>`, in place: the output follows the order of the input. Use
`always_prefix=True` to rename every key:

```pycon
>>> from coola.nested import flatten_mapping
>>> metrics = {"train": {"loss": 1, "acc": 2}, "val": {"loss": 3}}
>>> flatten_mapping(metrics, on_duplicate="prefix")
{'train.loss': 1, 'acc': 2, 'val.loss': 3}
>>> flatten_mapping(metrics, always_prefix=True, separator="/")
{'train/loss': 1, 'train/acc': 2, 'val/loss': 3}
>>> flatten_mapping(metrics, on_duplicate="last")
{'loss': 3, 'acc': 2}

```

With the default `on_duplicate="raise"`, a `DuplicateKeyError` is raised for the key `loss`.

## Flat Dictionaries

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

### Customizing the Separator

You can change the separator used to join keys:

```pycon
>>> from coola.nested import to_flat_dict
>>> nested = {"a": {"b": {"c": 1}}}
>>> to_flat_dict(nested, separator="/")
{'a/b/c': 1}

```

### Handling Lists and Tuples

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

### Mixed Nested Structures

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

`to_flat_dict()` enters the containers of `coola.iterator` (mappings, sequences, `deque`, named
tuples, dataclasses). Text, bytes, iterators, `range` and the unknown types are leaves. The sets are
leaves too, because a set has no index. A dataclass is entered field by field:

```pycon
>>> from dataclasses import dataclass
>>> from coola.nested import to_flat_dict
>>> @dataclass
... class Point:
...     x: int
...     y: int
>>> to_flat_dict({"p": Point(1, 2), "tags": {"a"}})
{'p.x': 1, 'p.y': 2, 'tags': {'a'}}

```

### Restoring Lists

`from_flat_dict()` keeps the integer segments as dictionary keys by default, so a list does not come
back as a list. Use `restore_lists=True` to rebuild the lists. A dictionary built from the keys whose
keys are exactly `"0"`, `"1"`, ..., `"n-1"` is converted, and the tuples come back as lists. A
dictionary given as a value is a value of the caller: it is kept as it is. There is no limit on the
depth:

```pycon
>>> from coola.nested import from_flat_dict
>>> flat = {"items.0": 10, "items.1": 20, "matrix.0.0": 1, "matrix.1.0": 3}
>>> from_flat_dict(flat)
{'items': {'0': 10, '1': 20}, 'matrix': {'0': {'0': 1}, '1': {'0': 3}}}
>>> from_flat_dict(flat, restore_lists=True)
{'items': [10, 20], 'matrix': [[1], [3]]}

```

### Converting Types to Strings

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

## JSON Conversion

Use `convert_to_jsonable()` to recursively convert a nested structure to JSON-compatible data.
Dataclasses and pydantic models found at any depth are converted to dictionaries, and sets are
converted to sorted lists (the order does not depend on the iteration order of the set, so it is the
same in every process). A dataclass is converted by the engine, field by field, like any other
container, and a tuple stays a tuple:

```pycon
>>> from dataclasses import dataclass
>>> from coola.nested import convert_to_jsonable
>>> @dataclass
... class Point:
...     x: int
...     y: int
>>> convert_to_jsonable([Point(x=1, y=2), {"key": Point(x=3, y=4)}, {3, 1, 2}])
[{'x': 1, 'y': 2}, {'key': {'x': 3, 'y': 4}}, [1, 2, 3]]

```

## Flattening a polars DataFrame

Use `flatten_frame()` to unnest the struct columns of a polars `DataFrame` (and, with
`expand_lists=True`, to expand the list columns into one column per index). The new columns are named
`<parent><separator><field>`. It requires `polars`, which is imported on first use:

```pycon
>>> import polars as pl
>>> from coola.nested import flatten_frame
>>> frame = pl.DataFrame({"id": [1, 2], "coords": [{"x": 10, "y": 20}, {"x": 30, "y": 40}]})
>>> flatten_frame(frame).columns
['id', 'coords.x', 'coords.y']

```

A `LazyFrame` is not accepted by `flatten_frame`: it raises a `TypeError` that asks you to call
`collect()` first, because a lazy frame has no cheap schema and the number of rows of an exploded list
is only known after the evaluation. The supported range of polars is `polars>=1.0,<3.0`.

## What `coola.nested` does not do

- **Keys of `convert_to_jsonable`.** The output is accepted by the Python `json` module, not by a
  strict JSON parser: the keys are not converted to strings, and `NaN` and the infinities are kept.
- **Dataclasses and pydantic models from `from_flat_dict`.** The function builds dicts and lists:
  rebuilding a typed object needs a schema.
- **A deep merge.** `merge_mappings` merges the top level with a duplicate strategy; it does not
  merge nested mappings.
- **A rename function other than a prefix and a suffix.** `add_prefix_suffix_to_keys` is the only
  public rename.
- **Lazy or parallel application.** The outputs are built whole, as in `coola.recursive`.

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

## Properties

The functions fall into five groups: `flat` (`to_flat_dict`, `from_flat_dict`, `flatten_frame`),
`keys` (`remove_keys_if`, `remove_keys_containing`, `remove_keys_starting_with`,
`add_prefix_suffix_to_keys`), `mappings` (`merge_mappings`, `flatten_mapping`, `DuplicateKeyError`),
`records` (`convert_to_dict_of_lists`, `convert_to_list_of_dicts`) and `json`
(`convert_to_jsonable`). The table says which group each property covers.

| Property                | Statement                                                                                                                                                                                                                                              | Groups                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| Pure                    | The input is never modified. A function that rebuilds containers returns new ones (the leaves and the values are shared, never copied). No function keeps state between calls.                                                                         | all                    |
| One notion of container | A function that looks inside the values enters the containers of the other packages (sequences, mappings, sets, dataclasses) and treats text, bytes, iterators and unknown types as leaves. `to_flat_dict` does not enter sets.                        | `flat`, `keys`, `json` |
| Structure-preserving    | A function that rebuilds a structure keeps the type of every container, the top one included, and the order of the entries.                                                                                                                            | `keys`, `json`         |
| Any depth               | No `RecursionError`, whatever the nesting.                                                                                                                                                                                                             | `flat`, `keys`, `json` |
| Cycle-safe              | A container that contains itself raises `ValueError`, never `RecursionError`.                                                                                                                                                                          | `flat`, `keys`, `json` |
| Keys are labels         | Only the `flat` and `keys` functions read or change keys. `add_prefix_suffix_to_keys` renames the `str` keys and keeps the others; `to_flat_dict` writes every key with `str()`.                                                                       | `flat`, `keys`         |
| No silent loss          | Two entries are never merged into one without a decision of the caller: a key collision raises (`DuplicateKeyError` or `ValueError`). A wrong input type raises `TypeError` naming the type.                                                           | all                    |
| Deterministic           | The same input gives the same output in every process: the entries keep the order of the input; a set becomes a sorted list.                                                                                                                           | all                    |
| Documented round trips  | `from_flat_dict(to_flat_dict(x)) == x` for `dict`/`str` keys that do not contain the separator (or with `escape=True`); with `restore_lists=True`, also for lists. `convert_to_list_of_dicts(convert_to_dict_of_lists(r)) == r` for non-empty records. | `flat`, `records`      |
| Explicit options        | Options are keyword-only. An invalid option raises `ValueError` before any work is done.                                                                                                                                                               | all                    |
| Lazy                    | `import coola.nested` imports no optional library (polars is loaded by the first use of `flatten_frame`).                                                                                                                                              | all                    |
| Thread-safe             | The state of a call is local to the call. There is no module-level mutable state.                                                                                                                                                                      | all                    |
| Fast                    | The cost is linear in the number of nodes.                                                                                                                                                                                                             | all                    |

Not guaranteed:

- **Shared references.** A container reachable by several paths is visited once per path, so
  `to_flat_dict` of `[x, x]` has the entries of `x` twice.
- **Dataclasses do not come back.** `from_flat_dict` gives dictionaries: the round trip is for dicts
  and lists only.
- **The type of a list-like.** `to_flat_dict` gives the same entries for a list and a tuple, so
  `from_flat_dict(..., restore_lists=True)` returns lists.
- **Non-`str` keys of `to_flat_dict`.** They are written with `str()` and cannot be restored.
- **Strict JSON.** `convert_to_jsonable` follows the Python `json` module: `NaN` is kept, the keys are
  not converted, tuples stay tuples.

## See Also

- [`coola.recursive`](recursive.md): For recursively transforming nested data
- [`coola.iterator`](iterator.md): For iterating over nested data structures
- [`coola.equality`](equality.md): For comparing nested data structures
