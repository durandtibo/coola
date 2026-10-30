# Converting Data to JSON

:book: This page describes the `coola.jsonable` package, which provides `to_jsonable`, a function
that converts nested data (dataclasses, sets, paths, arrays, pydantic models, ...) to JSON data.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

`to_jsonable` turns any nested data into **JSON data**: the values that `json.loads` returns, that
is `dict` with `str` keys, `list`, `str`, `int`, `float`, `bool` and `None`.

```pycon
>>> from dataclasses import dataclass
>>> from pathlib import PurePosixPath
>>> from coola.jsonable import to_jsonable
>>> @dataclass
... class Point:
...     x: int
...     y: int
>>> to_jsonable({"points": [Point(x=1, y=2)], "path": PurePosixPath("a/b"), 3: (1, 2), "s": {3, 1}})
{'points': [{'x': 1, 'y': 2}], 'path': 'a/b', '3': [1, 2], 's': [1, 3]}

```

The conversion is chosen from the type of each value, along the Method Resolution Order. The
containers are entered at any depth, and the optional libraries (NumPy, PyTorch, pandas, pydantic) are
imported only when a value of that library is converted.

| Type                                                      | JSON value                                                  |
| --------------------------------------------------------- | ----------------------------------------------------------- |
| `str`, `int`, `float`, `bool`, `None`                     | kept (a subclass, such as an `IntEnum` member, is kept)     |
| `list`, `tuple`, `deque`, `range`, other `Sequence`       | `list`                                                      |
| `set`, `frozenset`                                        | sorted `list`                                               |
| `dict`, `Mapping`, `MappingProxyType`                     | `dict` with `str` keys                                      |
| dataclass instance                                        | `dict` of the fields                                        |
| `PurePath`, `UUID`, `Decimal`, `UserString`               | `str`                                                       |
| `bytes`, `bytearray`                                      | Base64 `str`                                                |
| `datetime.date`, `datetime.datetime`, `datetime.time`     | ISO 8601 `str`                                              |
| `datetime.timedelta`                                      | number of seconds (`float`)                                 |
| `enum.Enum`                                               | its value, converted                                        |
| `numpy.ndarray`, NumPy scalar                             | nested `list`, Python value                                 |
| `torch.Tensor`                                            | nested `list`                                               |
| `pandas.Series`, `pandas.Index`; `pd.NaT`, `pd.NA`        | `list`; `None`                                              |
| `pydantic.BaseModel`; `SecretStr`, `SecretBytes`          | `dict` of `model_dump()`, converted; `"**********"`         |

## Properties

| Property                 | Statement                                                                                                                                                                                                                                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JSON data model          | The output contains only `dict` with `str` keys, `list`, `str`, `int`, `float`, `bool` and `None`. Tuples, deques, sets and ranges become lists, dataclasses and pydantic models become dicts. A subclass of `str`, `int` or `float` (an `IntEnum` member) is kept: `json` writes it as its base value. |
| Accepted by `json`       | With `on_unsupported="raise"` (the default), `json.dumps(to_jsonable(x))` never raises, and `json.loads` of it equals `to_jsonable(x)` (`NaN` aside). The function is idempotent: `to_jsonable(to_jsonable(x)) == to_jsonable(x)`.                                                                       |
| Strict and loud          | A value whose type has no transformer raises a `TypeError` that names the type and tells how to register one. It is never silently returned unchanged, never replaced by `repr` or `None`. `"warn"` and `"keep"` are explicit choices of the caller.                                                      |
| Type dispatch            | The transformer is chosen from `type(data)` along the Method Resolution Order, so the most specific registered type wins.                                                                                                                                                                                 |
| Registry-extensible      | A new type is supported by registering one transformer, without changing another one. The plugin group `coola.jsonable.transformers` does it from another package.                                                                                                                                       |
| Keys are strings         | A `str` key is kept; `bool`, `int`, `float` and `None` are written like `json.dumps` writes them (`"true"`, `"1"`, `"1.5"`, `"null"`); any other key is converted by the registry (a `Path`, a `UUID`, an `Enum`) and must give one of these. A tuple key raises `TypeError`. Two keys that give the same string raise `ValueError`. |
| No silent loss           | Two entries are never merged into one, a value is never dropped and a secret is never written: `SecretStr` and `SecretBytes` give `"**********"`.                                                                                                                                                         |
| Deterministic            | The same input gives the same output in every process, whatever `PYTHONHASHSEED` is: dictionaries keep the order of the input and a set becomes a sorted list.                                                                                                                                            |
| Pure                     | The input is never modified. The output shares no container with the input. A registry is read, never modified by a call, so it can be shared by threads.                                                                                                                                                |
| Cycle-safe               | A container that contains itself raises `ValueError`.                                                                                                                                                                                                                                                     |
| Any depth                | No `RecursionError`, whatever the nesting (`json.dumps` itself raises one at about 1,000 levels).                                                                                                                                                                                                         |
| Consistent with siblings | The same containers as `equality`, `hashing`, `summary`, `recursive` and `iterator`: dataclasses are entered, text and bytes are leaves, iterators and generators are not accepted. Options are keyword-only.                                                                                            |
| Lazy                     | `import coola.jsonable` imports no optional library.                                                                                                                                                                                                                                                      |
| Fast                     | The cost is linear in the number of nodes, and there is no per-call setup.                                                                                                                                                                                                                                |

Not guaranteed:

- **A round trip to the Python types.** `json.loads` gives lists, not tuples, sets or dataclasses; a
  `Decimal`, a `UUID` or a `Path` becomes a `str`, bytes become a Base64 `str` (not distinguishable
  from text), a `timedelta` becomes a number of seconds. The conversion is one way.
- **Strict JSON (RFC 8259).** `NaN` and the infinities are kept, as `json.dumps` writes them by
  default. Pass `allow_nan=False` to `json.dumps` (or `save_json`) to refuse them.
- **Shared references.** A container reachable by several paths is converted once per path, so the
  output has no aliasing.
- **Nanosecond `datetime64` and `timedelta64`.** NumPy returns an `int`, which is kept; cast to a
  coarser unit first.
- **`numpy.longdouble`.** Its `tolist` keeps NumPy scalars, so it raises a `TypeError`; cast to
  `float64` first.
- **Tables.** Data frames, tables and datasets have no single JSON shape; they are not converted until
  you say how (see below).

## Unsupported values

`on_unsupported` says what happens to a value whose type has no transformer.

| Value       | Effect                                                                    |
| ----------- | ------------------------------------------------------------------------- |
| `"raise"`   | `TypeError` that names the type (default)                                 |
| `"warn"`    | `UserWarning` for each value, the value is kept                           |
| `"keep"`    | The value is kept silently                                                |

```pycon
>>> from coola.jsonable import to_jsonable
>>> class Obj:
...     pass
>>> to_jsonable([Obj()])
Traceback (most recent call last):
    ...
TypeError: Cannot convert an object of type Obj to JSON: register a ConvertTransformer for it with coola.jsonable.register_transformers, or pass on_unsupported='keep'
>>> to_jsonable([1, 1 + 2j], on_unsupported="keep")
[1, (1+2j)]

```

With `"warn"` and `"keep"` the result may contain objects that `json` rejects: you own them. A key that
cannot become a string raises `TypeError` in the three modes, and an invalid `on_unsupported` raises
`ValueError` before any work is done.

## Supporting a new type

Register a `ConvertTransformer`: a function that maps a value to a simpler one. Its result is
converted in turn, so it may return a container, a path or an array. A function must reduce the type to
simpler ones: a function that returns a value of the same type raises a `TypeError`, and a cycle of
types (A gives B, B gives A) never ends.

```pycon
>>> from coola.jsonable import ConvertTransformer, register_transformers, to_jsonable
>>> class Point:
...     def __init__(self, x, y):
...         self.x, self.y = x, y
>>> register_transformers({Point: ConvertTransformer(lambda p: {"x": p.x, "y": p.y})})
>>> to_jsonable([Point(1, 2)])
[{'x': 1, 'y': 2}]

```

`register_transformers` extends the default registry, which is shared by the process. To keep the
change local, copy the registry and pass it to `to_jsonable` (`exist_ok=True` replaces a transformer
that is already registered):

```pycon
>>> from coola.jsonable import get_default_registry
>>> registry = get_default_registry().copy()
>>> registry.register(Point, ConvertTransformer(lambda p: [p.x, p.y]), exist_ok=True)
>>> to_jsonable(Point(1, 2), registry=registry)
[1, 2]

```

A package can register its types without a call from the user with the entry-point group
`coola.jsonable.transformers` (see [Extending coola](extending.md)).

### Frames and tables

A data frame has several usual shapes (a list of records, a dict of columns, ...) and none is neutral,
so it is not converted until you choose one:

```python
import pandas as pd
from coola.jsonable import ConvertTransformer, register_transformers

register_transformers({pd.DataFrame: ConvertTransformer(lambda df: df.to_dict(orient="records"))})
```

The same holds for polars, pyarrow and xarray objects.

### Pydantic models

A model is dumped with `model_dump()` (Python mode) and converted by the registry, so the sets inside
are sorted and the arrays are converted. A serializer declared with `when_used="json"` is therefore
not applied. A model that needs its JSON mode registers
`ConvertTransformer(lambda model: model.model_dump(mode="json"))` for its class.

## Writing a JSON file

`save_json` writes what it receives, so convert first:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> import numpy as np
>>> from coola.io import load_json, save_json
>>> from coola.jsonable import to_jsonable
>>> batch = {"x": np.arange(3), "path": Path("a"), "ids": {2, 1}}
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("batch.json")
...     save_json(to_jsonable(batch), path)
...     load_json(path)
{'x': [0, 1, 2], 'path': 'a', 'ids': [1, 2]}

```

## What `coola.jsonable` does not do

- **Read back.** There is no `from_jsonable`, type tag or schema: the conversion is one way. Use
  pydantic or `msgspec` to validate the data you read.
- **Other formats** (YAML, TOML, MessagePack). The decisions on bytes, durations and dates are
  specific to JSON.
- **Streaming or JSON Lines.** The output is built whole. A generator is not accepted.
- **Formatting the file** (indentation, `ensure_ascii`, sorted keys): the job of `json.dumps` and
  `JsonSaver(**kwargs)`.
- **Big integers beyond 2^53.** A JSON reader in another language may lose them: the output keeps the
  Python `int`.

## See Also

- [`coola.recursive`](recursive.md): The engine that walks the structure
- [`coola.nested`](nested.md): Flat dictionaries, keys and records
- [`coola.io`](io.md): Loaders and savers
- [Extending coola](extending.md): Plugins and entry points
