# Architecture of `coola.jsonable`

:book: This page describes the design of the `coola.jsonable` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/jsonable.md#properties); the API is in the [reference](../refs/jsonable.md). The analysis that led to this design is in `review/json.md`.

## Overview

`coola.jsonable` is **a registry of transformers on the engine of `coola.recursive`**. It adds no
traversal: the cycles, the depth, the order, the per-call cache and the thread safety come from the
engine. A transformer only describes the shape of one type with the `expand` contract of `recursive`:
`expand(data)` returns `None` for a scalar, or `(children, rebuild)`. Three shapes are enough for every
type:

| Shape          | `expand(data)`                                                       | Used for                                       |
| -------------- | -------------------------------------------------------------------- | ---------------------------------------------- |
| scalar         | `None`: `str`, `int`, `float`, `bool`, `None` are kept               | the JSON scalars (`DefaultTransformer`)        |
| container      | `(children, rebuild)`: the rebuild makes a `list` or a `dict`        | sequences, sets, mappings, dataclasses         |
| **conversion** | `([convert(data)], lambda xs: xs[0])`: the result is converted again | every other type (`Path`, `Enum`, arrays, ...) |

The third shape is the whole trick (`ConvertTransformer`). A conversion returns a value that the
engine converts in turn, until only JSON scalars and containers remain: an `Enum` gives its value
(which may be a tuple of paths), `ndarray.tolist()` gives a list that is checked like any list, a
pydantic model gives a dict whose values may be sets or arrays, a `SecretStr` gives a `str`. No
converter calls another one, no converter knows the registry, and nothing is returned unchecked.

## Constraints

1. **Simple and easy to maintain.** One function, one registry, one rule per type. The old table of
   predicates, the `sys.modules` look-ups and the registry built on every call are deleted.
2. **Major release.** No alias for `coola.nested.convert_to_jsonable` or
   `coola.nested.conversion.to_jsonable`.
3. **No slower than before.** The benchmark gate (`tests/benchmarks/test_jsonable_benchmark.py`) holds
   on every input of the design, including arrays (see D12 below).
4. **Consistent with the siblings.** Same containers, same dispatch (`TransformerRegistry`), same lazy
   loading (`register_lazy`), same plugin mechanism (`make_default_registry_singleton` with a
   `plugin_group`), same keyword-only options.
5. **Optional libraries stay lazy.** NumPy, PyTorch, pandas and pydantic are imported by their module
   only, never by `import coola.jsonable`.

## Architecture in detail

```
coola/jsonable/
    __init__.py       package docstring with the Properties; the public names
    interface.py      to_jsonable, get_default_registry, register_transformers; the default registry
    transformers.py   ConvertTransformer, SequenceTransformer, SetTransformer, MappingTransformer,
                      DataclassTransformer
    stdlib.py         get_default_transformers(): the standard library types
    numpy.py          get_default_transformers(): ndarray (NDArrayTransformer), generic
    torch.py          get_default_transformers(): Tensor (TensorTransformer)
    pandas.py         get_default_transformers(): Series, Index, NaT, NA
    pydantic.py       get_default_transformers(): BaseModel, SecretStr, SecretBytes
```

```
to_jsonable(data, *, registry=None, on_unsupported="raise")
        |   one registry (default: the singleton), one call = one engine run
        v
registry.transform(data, leaf)                 coola.recursive engine
        |-- transformer = registry.find_transformer(type(data))   MRO + fallback + lazy libraries
        |-- expansion   = transformer.expand(data)
        |        None                      -> a leaf: leaf(data)
        |        (children, rebuild)       -> the engine converts the children, then rebuild(children)
        v
JSON data
```

`interface.py` follows `summary/interface.py`: `_register_default_transformers(registry)` registers the
native and standard types, `register_fallback(dataclasses.is_dataclass, DataclassTransformer())` and
one `register_lazy` per installed library, and the singleton is built with
`make_default_registry_singleton(..., plugin_group="coola.jsonable.transformers")`.

### The leaf function and `on_unsupported`

`object` and the JSON scalars are registered with `DefaultTransformer`, so a value of an unregistered
type reaches the **leaf function** of the engine. The leaf function is the only place where the three
modes differ: `_raise` keeps the JSON scalars and raises a `TypeError` for the others, `_warn` warns
and keeps, `_keep` is the identity. The transformers are the same in the three modes, so there is one
registry and nothing is built per call. The check is one `isinstance` against the five scalar types
(about 0.03 µs per leaf).

### Keys

The **keys are children** of the mapping (`[*keys, *values]`), so the engine converts them like any
other value (a `Path` key gives a `str`, a tuple key gives a list). `_to_dict` then checks the result:
a `str` is kept, `bool`, `int`, `float` and `None` go through `json.dumps` (`True` gives `"true"`), any
other value raises a `TypeError` that names the type, and a duplicate raises a `ValueError` that names
the key. This is the only place where "keys are labels" of `recursive` is deliberately reversed,
because a JSON object has no other kind of key.

### Traps of the dispatch

- `UserString` is a `Sequence` of `UserString`: a `Sequence` transformer that lists its elements never
  ends. It is registered as a `str` conversion in `stdlib.py`.
- `numpy.generic` comes before `float` in the MRO of `numpy.float64`, so a NumPy scalar is converted by
  `item()`. `np.bytes_` is a `bytes` subclass and `np.str_` a `str` subclass: they follow `bytes` and
  `str`.
- A `str` enum member or an `IntEnum` member is a `str` or an `int` for the MRO, so it is a scalar and
  is kept (D11). Register the enum class to normalize it.
- `pd.NaT` is a `datetime` subclass: `type(pd.NaT)` is registered, so it gives `None` and not
  `"NaT"`.
- A dataclass class (not an instance) has the type `type`, so it is a leaf and is unsupported.
- `ConvertTransformer` rejects a function that returns a value of the type of its input (it would
  never end). It cannot see a cycle of types: the contract test of the default registry and the rule
  "a conversion must reduce the type to simpler ones" cover it (D13).

## Decisions

- **D1. The output is the JSON data model** (tuples become lists), which gives the idempotence
  property.
- **D2. Unsupported types: three modes**, `"raise"` (default), `"warn"` and `"keep"`. Rejected: a
  `default=` callable like `json.dumps` (the registry already is the per-type hook), `repr` as a mode
  (`ConvertTransformer(repr)` registered for `object` in a copy of the registry does it).
- **D3. Pydantic: `model_dump()` in Python mode, then the registry.** The sets are sorted, the arrays
  and paths are converted, a secret is masked, and the exclusions and plain serializers of the model
  are honoured. A serializer declared with `when_used="json"` is not applied. Rejected: `mode="json"`
  (non-deterministic sets, fails on a NumPy field); the fields as children (ignores `exclude` and
  every serializer).
- **D4. `NaN` and the infinities are kept.** `json.dumps(..., allow_nan=False)` enforces RFC 8259.
- **D5. Bytes are Base64.** Lossless but indistinguishable from text when read.
- **D6. A `timedelta` is a number of seconds.** A `float` loses the microseconds above about 285
  years.
- **D7. Frames and tables are not converted by default.** A `DataFrame` has at least three usual
  shapes and none is neutral. `Series` and `Index` are one-dimensional and are converted.
- **D8. A new package `coola.jsonable`, one function.** The name avoids `coola.json`, which hides the
  standard library.
- **D9. `register_transformers` and `ConvertTransformer`** rather than a function-based API: the
  registry is the extension point of every package.
- **D10. The keys go through the registry**, so a `Path`, a `UUID` or an `Enum` key works with no code
  in the mapping transformer.
- **D11. The subclasses of the JSON scalars are kept.** Normalizing them would need an exact-type check
  on every leaf and would not change what `json` writes.
- **D12. The output of an array is checked, except for the numeric dtypes.** Checking every element
  makes complex numbers, object arrays and `datetime64` raise (they are not JSON), but costs 0.09 µs
  per element: the conversion of 1,000,000 floats was 149 ms, 7.9 times `tolist` and 47 % of
  `json.dumps` of the result, which failed the benchmark gate. `NDArrayTransformer` and
  `TensorTransformer` therefore return no children for the arrays whose `tolist` already gives Python
  `bool`, `int` and `float` (NumPy kinds `b`, `i`, `u` and the dtypes `float16`, `float32`, `float64`;
  every real tensor): 17 ms, the cost of `tolist`. The other arrays (complex, object, dates, strings,
  `longdouble`) are still checked.
- **D13. No guard against a converter that never ends**, other than its own type. The engine has no
  depth limit by design.

## Public API

```python
from coola.jsonable import (
    to_jsonable,  # (data, *, registry=None, on_unsupported="raise")
    get_default_registry,  # the singleton TransformerRegistry
    register_transformers,  # (mapping, *, exist_ok=False)
    ConvertTransformer,  # (func): the building block of a custom type
    DataclassTransformer,
    MappingTransformer,
    SequenceTransformer,
    SetTransformer,
)
```

## Dependents

| Package or file                  | Relation                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `coola.recursive`                | `TransformerRegistry`, `BaseTransformer`, `DefaultTransformer` and the engine are reused as they are. |
| `coola.nested`                   | `convert_to_jsonable` and `conversion.to_jsonable` are removed; `conversion.py` is now `records.py`.  |
| `coola.io` (`save_json`)         | The docstring says "use `coola.jsonable.to_jsonable` first". No code change.                          |
| `pyproject.toml` (import-linter) | `coola.jsonable` is in the lists of the contracts: the core packages must not import it.              |
| Entry points                     | New plugin group `coola.jsonable.transformers`.                                                       |

## Tests

The unit tests mirror the source tree (`tests/unit/jsonable/`). The property tests
(`tests/properties/test_jsonable.py`) check that the output is accepted by `json` and round trips, that
the function is idempotent, that the output does not depend on the order of the input and that the
input is not modified. The integration tests (`tests/integration/jsonable/`) check the output for five
values of `PYTHONHASHSEED` (with a pydantic model that has a set), the threads, the depth of 100,000
levels and the cycles. The lazy-import tests check that `import coola.jsonable` imports no optional
library. The benchmark (`tests/benchmarks/test_jsonable_benchmark.py`) is the gate for the speed.
