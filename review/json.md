# `coola.jsonable`: target state and the changes to reach it

This document describes the design that the JSON conversion of `coola` should have, then the changes
to make to the current implementation (`main`, `9659de0b`) to get there. Today the conversion is
`coola.nested.convert_to_jsonable` plus `coola.nested.conversion.to_jsonable` (225 of the 352 lines of
`src/coola/nested/conversion.py`, 48 of its tests). The target is a small sub-package,
**`coola.jsonable`**, with **one function (`to_jsonable`), one registry, one rule per type**. The
design follows the rules of the user guides (`docs/docs/uguide/recursive.md`, `nested_data.md`) and the
architecture guide (`docs/docs/dev/architecture.md`: properties first, one engine, one extension
point, strict and loud, lazy imports), and `review/recursive.md` is the model. Breaking changes are
accepted, and the future state is more important than the past one.

Evidence comes from probes (about 60 inputs) and a throw-away prototype run on `main` (Python 3.14,
NumPy, PyTorch, pandas, pydantic; scripts kept outside the repository). Items marked **[verified]**
were reproduced; **[reading]** comes from the code. Nothing in `src`, `tests` or `docs` was changed.

**Status:** implemented in four pull requests: step 1 #2652, step 2 #2653, step 3 #2654, step 4 #2656. One deviation: the benchmark gate of section 8 failed for the array with the check of every element (149 ms, 7.9x `tolist`, 47 % of `json.dumps`), so the fast path that D12 names is included (`NDArrayTransformer`, `TensorTransformer`: no check of the elements of real numeric arrays, 17 ms).

## 1. Properties of the sub-package

`to_jsonable(data)` turns any nested data into **JSON data**: the values that `json.loads` returns.
These are the rules. Every design choice below exists to make one of them true **by construction**
(in the engine, in one transformer) instead of by the discipline of each converter.

| Property                 | Statement                                                                                                                                                                                                                                                                                                                                                                                                                    | Enforced by                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| JSON data model          | The output contains only `dict` with `str` keys, `list`, `str`, `int`, `float`, `bool` and `None`. Tuples, deques, sets and ranges become lists, dataclasses and pydantic models become dicts. A subclass of `str`, `int` or `float` (an `IntEnum` member) is kept: `json` writes it as its base value.                                                                                                                      | the registry (one transformer per type)    |
| Accepted by `json`       | With the default `on_unsupported="raise"` (and with `"warn"` when no warning was issued), `json.dumps(to_jsonable(x))` never raises, and `json.loads` of it equals `to_jsonable(x)` (`NaN` aside). Hence `to_jsonable(to_jsonable(x)) == to_jsonable(x)`: the function is idempotent.                                                                                                                                        | the property test (section 8)              |
| Strict and loud          | By default (`on_unsupported="raise"`), a value whose type has no transformer raises a `TypeError` that names the type and tells how to register one. It is never silently returned unchanged, never replaced by `repr` or `None`. The caller can choose `"warn"` (a `UserWarning` per value, the value is kept) or `"keep"` (the value is kept): both are explicit, and only they give up the property "Accepted by `json`". | the leaf function of the mode              |
| Type dispatch            | The transformer is chosen from `type(data)` along the Method Resolution Order, so the most specific registered type wins (the dispatch of every registry of `coola`).                                                                                                                                                                                                                                                        | `TransformerRegistry`                      |
| Registry-extensible      | A new type is supported by registering one transformer, without changing another one. A plugin group (`coola.jsonable.transformers`) does it from another package.                                                                                                                                                                                                                                                           | the registry, the plugin group             |
| Keys are strings         | Every key of the output is a `str`. A `str` key is kept; `bool`, `int`, `float` and `None` are written like `json.dumps` writes them (`"true"`, `"1"`, `"1.5"`, `"null"`); any other key is converted by the registry (a `Path`, a `UUID`, an `Enum`) and must give one of these. A key that cannot (a tuple) raises `TypeError`. Two keys that give the same string raise `ValueError`.                                     | the mapping transformer                    |
| No silent loss           | Two entries are never merged into one, a value is never dropped and a secret is never written: `SecretStr` and `SecretBytes` give `"**********"`. Everything that is lossy is listed under "Not guaranteed".                                                                                                                                                                                                                 | the transformers, tested one by one        |
| Deterministic            | The same input gives the same output in every process (whatever `PYTHONHASHSEED` is): dictionaries keep the order of the input, a set becomes a sorted list, and the sort never depends on the iteration order of the set.                                                                                                                                                                                                   | the set transformer                        |
| Pure                     | The input is never modified. The output shares no container with the input (leaves are immutable). A converter keeps no state; a registry is read, never modified, so it can be shared by threads.                                                                                                                                                                                                                           | the engine, stateless transformers         |
| Cycle-safe               | A container that contains itself raises `ValueError`, whichever transformer is registered for it.                                                                                                                                                                                                                                                                                                                            | the engine of `coola.recursive`            |
| Any depth                | No `RecursionError`, whatever the nesting (100,000 levels work; `json.dumps` itself raises a `RecursionError` at that depth **[verified]**). The depth is bounded by memory only.                                                                                                                                                                                                                                            | the engine of `coola.recursive`            |
| Consistent with siblings | The same containers as `equality`, `hashing`, `summary`, `recursive` and `iterator`: dataclasses are entered, text and bytes are leaves, iterators and generators are not accepted. Options are keyword-only.                                                                                                                                                                                                                | the default registry                       |
| Lazy                     | `import coola.jsonable` imports no optional library. The converters of NumPy, PyTorch, pandas and pydantic are loaded the first time the registry meets a type of the library, with no look at `sys.modules`.                                                                                                                                                                                                                | `register_lazy`                            |
| Fast                     | The cost is linear in the number of nodes, about 0.1 µs per leaf, and there is no per-call setup (no registry is built per call).                                                                                                                                                                                                                                                                                            | the engine, the default registry singleton |

Not guaranteed, and documented as such:

- **A round trip to the Python types.** `json.loads` gives lists, not tuples, sets or dataclasses; a
  `Decimal`, a `UUID` or a `Path` becomes a `str`, bytes become a Base64 `str`, a `timedelta` becomes
  a number of seconds (`float`). The conversion is one way.
- **Strict JSON (RFC 8259).** `NaN` and the infinities are kept, exactly as `json.dumps` writes them
  by default. Pass `allow_nan=False` to `json.dumps` (or `save_json`) to refuse them.
- **Shared references.** A container reachable by several paths is converted once per path (the rule
  of `recursive`), so the output has no aliasing.
- **Nanosecond `datetime64`.** NumPy returns an `int` for `datetime64[ns]` (`tolist`), which is kept;
  cast to a coarser unit first.
- **Tables.** Data frames, tables and datasets have no single JSON shape; they are not converted
  until the caller says how (decision D7).

## Constraints

1. **Simple and easy to maintain.** The target deletes code that exists only to work around the
   design (the predicate table `_CONVERTERS`, the `sys.modules` look-ups, the registry built per call)
   instead of adding code next to it. The prototype is about 120 lines of code; with the docstrings of
   the repository the package is about 450 lines (against 225 today), because it adds what the
   current code lacks: a registry, a plugin point, key handling and four library adapters. Each
   module stays small and does one thing, as in the sibling packages.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias, no
   deprecation period, `convert_to_jsonable` is not kept. The release notes and the migration page
   list the changes.
3. **No slower than today.** The benchmark gate of section 8 must hold. The prototype is faster on every
   input of section 6 (1.3x to 41x), with one exception that is stated (arrays, D12).
4. **Consistent with the siblings.** Same containers, same dispatch, same lazy loading, same plugin
   mechanism (`make_default_registry_singleton` with a `plugin_group`), same keyword-only options.
5. **Optional libraries stay lazy.** NumPy, PyTorch, pandas and pydantic are imported by their module
   of the package only, never by `import coola.jsonable`.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Status         |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The result is not JSON, and nothing says so.** An unknown type is returned unchanged: `convert_to_jsonable(Obj())` returns the `Obj`, `1+2j`, `Fraction(1, 3)`, `memoryview`, `slice`, a function or a class come back as they are, and `json.dumps` fails later, far from the cause. The contract is "accepted by `json`" but a `deque` (kept, not accepted), a `pd.Series`, a `pd.NA` and a `datetime.timezone` break it.                                                                                                                                                                                  | **[verified]** |
| F2  | **Keys are never converted.** `{(1, 2): 1}`, `{Path("a"): 1}` and `{b"a": 1}` come back unchanged and fail in `json.dumps`; `{1: "a", "1": "b"}` comes back unchanged and `json.dumps` **writes the key `"1"` twice** (two keys, one JSON object that parsers read differently). An `Enum` key works only by luck (a `str` enum).                                                                                                                                                                                                                                                                              | **[verified]** |
| F3  | **The leaf converter is a table of predicates, not a registry.** `to_jsonable` scans eight `(predicate, converter)` pairs for every leaf (`_CONVERTERS`), with `sys.modules.get(...)` look-ups for the optional libraries. It does not follow the Method Resolution Order, cannot be extended without editing the table (the property "Registry-extensible" of every other package), is invisible to the plugin mechanism, and its order is a hidden rule. It costs 1.1 µs per leaf: `convert_to_jsonable` of a list of 100,000 `int` takes 111 ms, the engine alone 8 ms.                                     | **[verified]** |
| F4  | **A registry is built on every call.** `convert_to_jsonable` copies the default registry and registers three entries on each call, so a scalar costs 10.3 µs and a small dict 16 µs (the prototype: 0.25 µs and 2.3 µs).                                                                                                                                                                                                                                                                                                                                                                                       | **[verified]** |
| F5  | **The output of a converter is not converted again.** `ndarray.tolist()` is returned as it is, so what NumPy puts in the list stays: `np.array([1+2j])` gives `[(1+2j)]`, an object array gives the objects, `np.datetime64("2020-01-01")` gives a `datetime.date`, `np.timedelta64(5, "s")` a `timedelta`, `np.bytes_(b"a")` bytes, all rejected by `json.dumps`. An `Enum` whose value is a tuple that contains a `Path` returns the tuple with the `Path`. `torch.tensor([1+2j])` gives `[(1+2j)]`.                                                                                                         | **[verified]** |
| F6  | **Pydantic models are opaque.** The model is dumped with `model_dump(mode="json")`, which the engine cannot see into: a model with a NumPy field raises `PydanticSerializationError` (the registry of `coola` knows NumPy), and a `set` field is written **in iteration order**, so the output depends on `PYTHONHASHSEED` (`['d','e','a','b','c']`, `['b','e','c','d','a']`, `['e','b','c','d','a']` for seeds 1, 2 and 3). The property "Deterministic" of `review/nested.md` holds everywhere except inside a model.                                                                                        | **[verified]** |
| F7  | **Missing values become garbage or fail.** `pd.NaT` is a `datetime` subclass and gives the **string** `"NaT"`, which is a valid value for a text column; `pd.NA` is returned unchanged; `pd.Series` and `pd.Index` are returned unchanged. `np.datetime64("NaT")` has the same problem.                                                                                                                                                                                                                                                                                                                        | **[verified]** |
| F8  | **The structure is half preserved.** A tuple stays a tuple, a `deque` stays a `deque`, an `OrderedDict` stays an `OrderedDict`, a named tuple stays a named tuple. None of them exists in JSON, so "structure-preserving" is a property that cannot be used (a caller who wants the data model must convert again) and it is the reason why the result of `convert_to_jsonable` is not accepted by `json` for a `deque` (F1).                                                                                                                                                                                  | **[verified]** |
| F9  | **Two functions for one job.** `to_jsonable` converts one object and does not enter containers or dataclasses; `convert_to_jsonable` walks the structure. A reader has to know which one to call, each docstring sends the reader to the other one, and `to_jsonable([Path("a")])` returns the list unchanged. `to_jsonable` is exported by `coola.nested.conversion` but not by `coola.nested`.                                                                                                                                                                                                               | **[reading]**  |
| F10 | **The JSON conversion lives in `nested`.** `coola.nested` is "helpers to reshape, filter and convert nested structures", with a properties table written for the group `flat` and `keys`; its `json` group has its own dispatch, its own optional-library adapters (NumPy, PyTorch, pydantic) and needs a plugin point, like `hashing` and `summary`. `review/utils.md` (F13) already moved `to_jsonable` out of `utils` because it was "half of an API".                                                                                                                                                      | **[reading]**  |
| F11 | **`UserString` and other virtual sequences.** Not a defect today (a `UserString` is returned unchanged, and `json.dumps` then fails), but it shows the trap of the new design: `UserString` is a `Sequence` of one-character `UserString`s, so a `Sequence` transformer that lists its elements **never ends** (the explicit stack grows until memory is exhausted; the prototype hung until killed). `recursive` registers `UserString` as a leaf for this reason; the JSON registry must register it as a string.                                                                                            | **[verified]** |
| F12 | **Lossy conversions are not listed.** `np.datetime64` with a nanosecond unit gives an `int` (an instant becomes a number, `1577836800000000001`, **[verified]**), a `timedelta` gives `float` seconds (no unit, loses microseconds above ~285 years), `bytes` give Base64 (not distinguishable from text), a `Decimal` gives a `str`. The docstring lists the types but not what is lost, so the caller finds out when reading the file back.                                                                                                                                                                  | **[reading]**  |
| F13 | **Documentation drift.** The docstring of `convert_to_jsonable` promises "converting `pydantic.BaseModel` objects found at any depth" and "a tuple stays a tuple" (neither holds inside a model, F6, and neither is JSON, F8), the guide (`uguide/nested.md`, "JSON Conversion") and `dev/nested.md` repeat it, the migration page has a section 12.5 on determinism, `save_json` says "use `coola.nested.convert_to_jsonable` first", and the package docstring of `coola.nested` lists `json` as a group. There is no page for the registry, the extension point or the plugin group, because there is none. | **[reading]**  |

What is already right and must not regress: sets become **sorted** lists whose order never depends on
the iteration order of the set (`_sorted_elements`: by value, else by `(type name, repr)`, and a
`NaN` forces the second rule); dataclasses are entered and converted field by field (not
`dataclasses.asdict`, which deep-copies); `SecretStr` and `SecretBytes` are masked; the explicit stack
(no `RecursionError`, cycles are a `ValueError`); lazy optional libraries (no import of NumPy, PyTorch
or pydantic by `import coola.nested`); `bytes` as Base64, `Decimal`, `UUID` and `Path` as `str`,
dates and times as ISO 8601 strings, an `Enum` as its (converted) value, a `range` as a list.

## 3. Architecture

```
to_jsonable(data, *, registry=None)           coola/jsonable/interface.py
        |   one registry (default: the singleton), one call = one engine run
        v
registry.transform(data, identity)            coola.recursive engine (explicit stack, cycles, cache)
        |
        |-- transformer = registry.find_transformer(type(data))   MRO + fallback + lazy libraries
        |-- expansion   = transformer.expand(data)
        |        None                      -> a JSON scalar: kept   (identity is the leaf function)
        |        (children, rebuild)       -> the engine converts the children, then rebuild(children)
        v
JSON data
```

**The package is a registry of transformers on the engine of `coola.recursive`.** It adds no
traversal: cycles, depth, order, the per-call cache and the thread safety come from the engine, as
they do for `nested`. A transformer only describes the shape of one type, with the `expand` contract
of `review/recursive.md`: `expand(data)` returns `None` for a scalar, or `(children, rebuild)`. Three
shapes are enough for every type:

| Shape          | `expand(data)`                                                           | Used for                                       |
| -------------- | ------------------------------------------------------------------------ | ---------------------------------------------- |
| scalar         | `None`: `str`, `int`, `float`, `bool`, `None` are kept                   | the JSON scalars (`DefaultTransformer`)        |
| container      | `(children, rebuild)`: the rebuild makes a `list` or a `dict`            | sequences, sets, mappings, dataclasses         |
| **conversion** | `([convert(data)], lambda xs: xs[0])`: **the result is converted again** | every other type (`Path`, `Enum`, arrays, ...) |

The third shape is the whole trick. A conversion returns a value that the engine converts in turn,
until only JSON scalars and containers remain: an `Enum` gives its value (which may be a tuple of
paths), `ndarray.tolist()` gives a list that is checked like any list (F5), a pydantic model gives a
dict whose values may be sets or arrays (F6), a `SecretStr` gives a `str`. No converter calls another
one, no converter knows the registry, and nothing is returned unchecked. The leaf function of the
engine is chosen by the option `on_unsupported`, and it is the only place where the three modes
differ: `object` is a leaf like the JSON scalars, so a value of an unregistered type reaches the leaf
function, which keeps the JSON scalars and treats the others as unsupported (`"raise"`: `TypeError`,
`"warn"`: `UserWarning` and the value is kept, `"keep"`: the value is kept, the function is the
identity and costs nothing). The transformers are the same in the three modes, so there is one
registry and nothing is built per call. The check is one `isinstance` against the five scalar types,
+0.03 µs per leaf (7.7 ms to 10.6 ms for 100,000 `int` in the prototype).

### 3.1 Modules

```
coola/jsonable/
    __init__.py       package docstring with the Properties; the public names
    interface.py      to_jsonable, get_default_registry, register_transformers; the default registry
                      (native types, lazy libraries, plugin group)
    transformers.py   ConvertTransformer (the public building block) and the transformers of the
                      containers: Sequence, Set, Mapping, Dataclass (the leaf functions of `on_unsupported` are in
                      `interface.py`)
    stdlib.py         get_default_transformers(): the standard library types (Path, UUID, Decimal,
                      bytes, date/time, timedelta, Enum, range, UserString)
    numpy.py          get_default_transformers(): ndarray, np.generic
    torch.py          get_default_transformers(): Tensor
    pandas.py         get_default_transformers(): Series, Index, NaT, NA
    pydantic.py       get_default_transformers(): BaseModel, SecretStr, SecretBytes
```

`interface.py` follows `summary/interface.py`: `_register_default_transformers(registry)` registers the
native and standard types, `register_fallback(dataclasses.is_dataclass, DataclassTransformer())`,
and one `register_lazy` per installed library (`(("numpy",), "coola.jsonable.numpy")`, ...), and the
singleton is `make_default_registry_singleton(TransformerRegistry, _register_default_transformers,
plugin_group="coola.jsonable.transformers")`. `get_default_registry()` returns it, and
`reset_default_registries()` and `temporary_default_registries()` already handle it.

### 3.2 The transformers

```python
class ConvertTransformer(InlineDisplayMixin, BaseTransformer[T]):
    """The data is replaced by ``func(data)``, which is converted in turn."""

    def __init__(self, func: Callable[[T], Any]) -> None:
        self._func = func

    def expand(self, data):
        converted = self._func(data)
        if type(converted) is type(data):  # a converter that returns its own type never ends
            raise TypeError(f"The converter of {type(data).__qualname__} returned the same type")
        return [converted], _first


# ``object`` and the JSON scalars are registered with ``DefaultTransformer`` (leaves). The leaf
# function of ``on_unsupported`` is the only code that tells a scalar from an unsupported value.
_SCALARS = (str, int, float, bool, type(None))


def _keep(value):  # "keep": everything that reaches the leaf function is kept
    return value


def _raise(value):  # "raise"
    if isinstance(value, _SCALARS):
        return value
    raise TypeError(
        f"Cannot convert an object of type {type(value).__qualname__} to JSON: register a "
        f"ConvertTransformer for it with coola.jsonable.register_transformers, or pass "
        f"on_unsupported='keep'"
    )


def _warn(value):  # "warn": same message as a UserWarning, then the value is kept
    if not isinstance(value, _SCALARS):
        warnings.warn(
            f"Cannot convert an object of type {type(value).__qualname__} to JSON; kept as is",
            stacklevel=2,
        )
    return value


class SequenceTransformer(...):  # list, tuple, deque, Sequence, range
    def expand(self, data):
        return list(data), _identity  # the rebuild receives a new list


class SetTransformer(...):  # set, frozenset
    def expand(self, data):
        return list(data), _sorted_elements  # kept from nested.conversion


class MappingTransformer(...):  # dict, Mapping, MappingProxyType
    def expand(self, data):
        return [*data.keys(), *data.values()], _to_dict


class DataclassTransformer(...):  # the fallback of ``dataclasses.is_dataclass``
    def expand(self, data):
        names = [field.name for field in dataclasses.fields(data)]
        return [getattr(data, name) for name in names], lambda xs: dict(zip(names, xs, strict=True))
```

The **keys are children** of the mapping (`[*keys, *values]`). The engine converts them like any other
value (a `Path` key gives a `str`, a tuple key gives a list), and `_to_dict` only has to check the
result: a `str` is kept, `bool`, `int`, `float` and `None` go through `json.dumps` (so `True` gives
`"true"`, as the standard library does), anything else raises a `TypeError` that names the type; a
duplicate raises a `ValueError` that names the key. This is the only place where "keys are labels" of
`recursive` is deliberately reversed, because a JSON object has no other kind of key (F2).

The default registrations (the contract test of section 8 checks every row):

| Type                                              | Transformer                                 | Output                                     |
| ------------------------------------------------- | ------------------------------------------- | ------------------------------------------ |
| `object`                                          | `DefaultTransformer` (leaf)                 | the leaf function of `on_unsupported`      |
| `str`, `int`, `float`, `bool`, `NoneType`         | `DefaultTransformer` (scalar)               | kept                                       |
| `list`, `tuple`, `deque`, `Sequence`, `range`     | `SequenceTransformer`                       | `list`                                     |
| `set`, `frozenset`                                | `SetTransformer`                            | sorted `list`                              |
| `dict`, `Mapping`, `MappingProxyType`             | `MappingTransformer`                        | `dict` with `str` keys                     |
| dataclass instances (fallback)                    | `DataclassTransformer`                      | `dict` of the fields, in order             |
| `UserString`                                      | `ConvertTransformer(str)`                   | `str` (**must not be a `Sequence`**, F11)  |
| `PurePath`, `UUID`, `Decimal`                     | `ConvertTransformer(str)`                   | `str`                                      |
| `bytes`, `bytearray`                              | `ConvertTransformer(b64encode(...).decode)` | Base64 `str`                               |
| `datetime.date` (and `datetime`), `datetime.time` | `ConvertTransformer(isoformat)`             | ISO 8601 `str`                             |
| `datetime.timedelta`                              | `ConvertTransformer(total_seconds)`         | `float` seconds                            |
| `enum.Enum`                                       | `ConvertTransformer(lambda e: e.value)`     | the value, converted                       |
| `numpy.ndarray`, `numpy.generic`                  | `tolist`, `item`                            | converted again (F5)                       |
| `torch.Tensor`                                    | `tolist`                                    | converted again                            |
| `pandas.Series`, `pandas.Index`; `NaT`, `NA`      | `tolist`; `lambda _: None`                  | list; `None` (F7)                          |
| `pydantic.BaseModel`                              | `ConvertTransformer(model_dump)`            | the dict of `model_dump()`, converted (D3) |
| `pydantic.SecretStr`, `SecretBytes`               | `ConvertTransformer(str)`                   | `"**********"`                             |

Two details of the dispatch that the tests pin: `numpy.generic` comes before `float` in the MRO of
`numpy.float64`, so a NumPy scalar is converted by `item()` and not kept as a `numpy.float64`; and a
`str` enum member or an `IntEnum` member is a `str` or an `int` for the MRO, so it is a scalar and is
kept (D11).

## 4. Public API after the change

```python
from coola.jsonable import (
    to_jsonable,  # (data, *, registry=None, on_unsupported="raise")
    get_default_registry,  # the singleton TransformerRegistry
    register_transformers,  # (mapping, *, exist_ok=False)
    ConvertTransformer,  # (func): the building block of a custom type
)
```

`to_jsonable(data, *, registry=None, on_unsupported="raise")` is the one function. It replaces `convert_to_jsonable` and the
old `to_jsonable`; there is no alias. `registry` is a `coola.recursive.TransformerRegistry` (a copy of
`get_default_registry()` plus the caller's entries). `on_unsupported` says what happens to a value
whose type has no transformer: `"raise"` (default) raises `TypeError`, `"warn"` issues a `UserWarning`
and keeps the value, `"keep"` keeps the value silently; any other string raises `ValueError` before
any work is done (validated with `coola.validation`). A key that cannot become a string always
raises `TypeError` (whatever the mode), and `ValueError` is raised for a cycle and for two keys that
give the same string. With `"warn"` and `"keep"` the result may contain objects that `json` rejects;
the caller owns them.

The extension example of the guide is the shortest possible one:

```python
from coola.jsonable import ConvertTransformer, register_transformers, to_jsonable


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


register_transformers({Point: ConvertTransformer(lambda p: {"x": p.x, "y": p.y})})
to_jsonable([Point(1, 2)])  # [{'x': 1, 'y': 2}]
```

A frame becomes JSON when the caller chooses the shape (D7):

```python
register_transformers({pd.DataFrame: ConvertTransformer(lambda df: df.to_dict(orient="records"))})
```

Writing the file is the job of `coola.io`: `save_json(to_jsonable(batch), path)`. The saver stays a
pure format (it does not convert); its docstring points to `coola.jsonable.to_jsonable`.

## 5. Dependents

| Package or file                                                            | Change                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.nested`                                                             | `convert_to_jsonable` and `conversion.to_jsonable` are deleted with their helpers (`_CONVERTERS`, `_is_instance_of_loaded`, `_is_array`, `_sorted_elements` moves to `jsonable`). `conversion.py` keeps the two record functions and is renamed `records.py` (the name `nested.md` planned). The `json` group disappears from the docstring, the properties table and the guide; the "Deterministic" and "Cycle-safe" rows lose the `json` column. |
| `coola.recursive`                                                          | No change. `TransformerRegistry`, `BaseTransformer`, `DefaultTransformer` and the engine are reused as they are; `recursive` gets a dependent.                                                                                                                                                                                                                                                                                                     |
| `coola.io` (`save_json` docstring, `uguide/io.md`)                         | The docstring says "use `coola.jsonable.to_jsonable` first". No code change.                                                                                                                                                                                                                                                                                                                                                                       |
| `pyproject.toml` (import-linter)                                           | `coola.jsonable` is added to the six lists of the contracts (the core packages must not import it, `testing` must not be imported by it). It may import the core packages and `coola.recursive`, like `nested`.                                                                                                                                                                                                                                    |
| Entry points                                                               | New plugin group `coola.jsonable.transformers` (a plugin gets the registry, as for the other groups).                                                                                                                                                                                                                                                                                                                                              |
| `tests/properties/test_cross_module.py`, `test_nested.py`                  | The two properties of `convert_to_jsonable` move to `tests/properties/test_jsonable.py` and become the stronger ones of section 8.                                                                                                                                                                                                                                                                                                                 |
| `tests/benchmarks/test_nested_benchmark.py`                                | The `convert_to_jsonable` entries move to `test_jsonable_benchmark.py`.                                                                                                                                                                                                                                                                                                                                                                            |
| `tests/integration/nested/test_jsonable.py`, `recursive/test_dataclass.py` | Move to `tests/integration/jsonable/`.                                                                                                                                                                                                                                                                                                                                                                                                             |
| `tests/integration/test_default_registries.py`, `test_lazy_imports.py`     | New rows for the registry of `jsonable` and for "importing `coola.jsonable` imports no optional library".                                                                                                                                                                                                                                                                                                                                          |
| Docs                                                                       | New `uguide/jsonable.md`, `dev/jsonable.md`, `refs/jsonable.md`, the `mkdocs.yml` navigation; `dev/architecture.md` (the list of features, the package table, the tree); `uguide/nested.md` (the JSON section is replaced by a link), `uguide/nested_data.md`, `dev/nested.md`; the migration page.                                                                                                                                                |

## 6. Evidence: a prototype

A prototype of the design (the six transformers, the stdlib table, NumPy, PyTorch and pydantic, on the
real `TransformerRegistry` and the real engine of `main`) was run on `main`. It is not part of the
repository. **[verified]** Machine: Apple M1, Python 3.14, best of 3.

| Input                                                | `main`   | Prototype   | Speed-up  |
| ---------------------------------------------------- | -------- | ----------- | --------- |
| list of 100,000 `int`                                | 111.2 ms | 7.3 ms      | 15x       |
| list of 100,000 `str`                                | 111.6 ms | 7.4 ms      | 15x       |
| dict of 100,000 entries (the keys are converted too) | 113.8 ms | 24.4 ms     | 4.7x      |
| tree, depth 10, width 3 (59,049 leaves)              | 87.3 ms  | 19.6 ms     | 4.5x      |
| 10,000 dataclasses of two fields                     | 37.8 ms  | 13.3 ms     | 2.8x      |
| the scalar `1`, per call                             | 10.3 µs  | 0.25 µs     | 41x       |
| `{"a": 1, "b": [2, 3]}`, per call                    | 16.0 µs  | 2.3 µs      | 7x        |
| list nested 100,000 levels                           | 167 ms   | 130 ms      | 1.3x      |
| `ndarray` of 1,000,000 `float`                       | 17.6 ms  | **93.5 ms** | **0.19x** |

The array is the one regression, and it is the price of F5: `main` returns `tolist()` unchecked
(17.6 ms, the cost of `tolist`), the prototype checks each of the 1,000,000 floats (0.09 µs per
element). `json.dumps` of the list then takes 318 ms, so the whole pipeline (convert and dump) goes
from about 336 ms to 411 ms (23 % slower), and the unchecked list is not always JSON (complex numbers, objects, `datetime64`). Decision
D12 keeps the check and names the fast path if it is ever needed (skip the check for the `b`, `i`,
`u`, `f` dtypes).

Behaviour of the prototype on the probes of F1 to F11 **[verified]**:

| Input                                                                      | `main`                                     | Prototype                                             |
| -------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| `Obj()`, `1+2j`, `Fraction(1, 3)`, `slice`, `len`                          | returned unchanged                         | `TypeError: Cannot convert ... type Obj`              |
| `{(1, 2): 1}`                                                              | unchanged, `json.dumps` fails              | `TypeError` (key of type `list`)                      |
| `{1: "a", "1": "b"}`                                                       | unchanged, `json.dumps` writes `"1"` twice | `ValueError: Two keys give the same JSON key`         |
| `{Path("a"): 1}`, `{True: 1}`, `{None: 1}`                                 | unchanged / `json` writes them             | `{'a': 1}`, `{'true': 1}`, `{'null': 1}`              |
| `deque([1])`, `(1, 2)`, `OrderedDict(a=1)`                                 | kept, not accepted by `json` for the first | `[1]`, `[1, 2]`, `{'a': 1}` (plain)                   |
| `np.array([1+2j])`, object array                                           | `[(1+2j)]`, the objects                    | `TypeError` (complex, `Obj`)                          |
| `np.datetime64("2020-01-01")`, `np.timedelta64(5, "s")`, `np.bytes_(b"a")` | `date`, `timedelta`, `bytes`               | `'2020-01-01'`, `5.0`, `'YQ=='`                       |
| `Enum` with the value `(1, 2)`                                             | `(1, 2)`                                   | `[1, 2]`                                              |
| model with a NumPy field                                                   | `PydanticSerializationError`               | `{'a': [0, 1]}`                                       |
| model with a set field, seeds 1, 2, 3                                      | three different orders                     | `['a','b','c','d','e']` for all                       |
| model with a `SecretStr`                                                   | `'**********'`                             | `'**********'` (unchanged)                            |
| `pd.NaT`                                                                   | `'NaT'`                                    | `None` (with `pandas.py`; the prototype kept `'NaT'`) |
| `UserString("ab")` with a `Sequence` transformer                           | n/a                                        | **never ends**; with `UserString` registered: `'ab'`  |

Two limits of the prototype: it has no `pandas.py` (so the `NaT` row is the design, not a
measurement), and its leaf function is the identity (the mode `"keep"`; the check of the other modes was
measured separately, section 3), so a `str` enum member is kept as a member (D11).

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Breaking |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1    | **The package and the containers (F1, F2, F4, F8, F9, F11).** `coola/jsonable/` with `transformers.py`, `stdlib.py`, `interface.py` (default registry, `to_jsonable`, `register_transformers`, plugin group), the properties in the docstring, the import-linter entries. The old functions of `nested` stay for one step and are not used by the new code. Unit tests of every transformer and of every row of the table of section 3.2; integration tests (hash seeds, threads, depth). | no       |
| 2    | **The libraries (F5, F6, F7).** `numpy.py`, `torch.py`, `pandas.py`, `pydantic.py` with `register_lazy`; lazy-import and default-registry integration tests; the `NaT` and `NA` handling; the pydantic hash-seed regression test.                                                                                                                                                                                                                                                         | no       |
| 3    | **Remove the old code (F3, F9, F10).** Delete `convert_to_jsonable`, `conversion.to_jsonable` and their helpers from `nested`, rename `conversion.py` to `records.py`, move the tests, the property tests and the benchmarks, update the docstrings of `nested` and `io`. No alias.                                                                                                                                                                                                       | yes      |
| 4    | **Documentation (F13).** `uguide/jsonable.md` (properties, extension, the lossy conversions, the recipe with `save_json`), `dev/jsonable.md`, `refs/jsonable.md`, `mkdocs.yml`, `architecture.md`, `nested.md`, `nested_data.md`; the migration page (section 12.5 becomes a "moved" entry with the behaviour changes of the list below); `review/suggestions.md`.                                                                                                                        | no       |

The behaviour changes to list in the migration page (all in step 3, the first two are the ones that
break calling code):

1. `coola.nested.convert_to_jsonable(x)` and `coola.nested.conversion.to_jsonable(x)` become
   `coola.jsonable.to_jsonable(x)`. The new function enters containers and dataclasses, so
   `to_jsonable([Path("a")])` converts the path.
2. A value of an unknown type raises `TypeError` (it was returned unchanged). The remedies are
   `register_transformers` with a `ConvertTransformer`, or `on_unsupported="keep"` to get the old
   behaviour (`"warn"` to see which values are kept).
3. Tuples, deques, named tuples and ranges become lists; `OrderedDict`, `defaultdict` and `Counter`
   become dicts.
4. The keys become strings (as `json.dumps` writes them) or raise; two keys that give one string raise.
5. NumPy, PyTorch and pydantic outputs are checked: complex numbers and objects raise (by default), a `datetime64`
   with a unit coarser than the nanosecond gives an ISO string instead of a `date`, a `timedelta64`
   gives seconds, `np.bytes_` gives Base64.
6. A pydantic model is dumped with `model_dump()` (Python mode) and converted by the registry: the sets
   inside are sorted and the arrays are converted; a serializer declared `when_used="json"` is no
   longer applied (D3).
7. `pd.NaT` and `pd.NA` give `None`; `pd.Series` and `pd.Index` give lists.

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/jsonable/test_<module>.py`); the conventions of
`CLAUDE.md` apply (one behaviour per test, literal expected values, no local imports, a regression
test for every bug, checked to fail without the fix).

- **Each transformer** (`test_transformers.py`): `expand` asserts the whole output against a literal
  (children and rebuilt value), the typical, empty, single-element and duplicate cases; the input is
  not modified and the output shares no container with it. `MappingTransformer`: a `str`, `int`,
  `float`, `bool`, `None` and `Path` key, a tuple key (`TypeError` with the type), two keys that give
  one string (`ValueError` with the key), `OrderedDict` and `defaultdict` give a `dict`, the order is
  kept. `SetTransformer`: the sort by value, mixed types, `NaN`, a set of frozensets, the move of the
  tests of `_sorted_elements` and `_sort_key`. `ConvertTransformer`: the result is converted again,
  a converter that returns its own type raises.
- **The default registry** (`test_interface.py`, new): one row per line of the table of section 3.2,
  with a sample and a literal expected value (like `NATIVE_SAMPLES` in `equality`), `UserString` is a
  string, `numpy.float64` goes through `item`, an `IntEnum` is kept, an unknown type raises in the
  default mode, warns and is kept with `"warn"`, is kept silently with `"keep"`, an invalid
  `on_unsupported` raises `ValueError`; an unknown type in a key raises in the three modes; the
  `TypeError` names the type; a dataclass class (not an instance) is unsupported; and `to_jsonable`
  does not modify the default registry.
- **The libraries** (`test_numpy.py`, `test_torch.py`, `test_pandas.py`, `test_pydantic.py`): with the
  `*_available` fixtures; the cases of F5 to F7 (complex, object array, `datetime64[D]`, a tensor with
  `requires_grad`, `bfloat16`, `NaT`, a model with a set, a NumPy field and a `SecretStr`).
- **Properties** (`tests/properties/test_jsonable.py`, replaces the two properties of
  `convert_to_jsonable`):
  1. for every nested value of the supported types (dict, list, tuple, set, dataclass, enum, path,
     `Decimal`, `bytes`, dates, NumPy scalars; a keys strategy that includes `int`, `bool`, `None`),
     `json.loads(json.dumps(to_jsonable(x), allow_nan=...))` equals `to_jsonable(x)` (no `NaN`);
  2. `to_jsonable(to_jsonable(x)) == to_jsonable(x)` (idempotent);
  3. the output does not depend on the order of the input sets (kept from `test_nested.py`);
  4. the input is not modified (`deepcopy` before, `objects_are_equal` after).
- **Integration** (`tests/integration/jsonable/`): the output for `PYTHONHASHSEED` 0, 1, 2, 3, 42 is the
  same, with a model that has a set (the regression test of F6, in a subprocess); threads running
  `to_jsonable` on the default registry with a lock around the shared list (`run_threads`, docstring
  "Test that ..."); depth 100,000 does not raise; a cycle through a list, a dict and a dataclass
  raises `ValueError`; importing `coola.jsonable` imports neither NumPy, PyTorch, pandas nor pydantic;
  a plugin registered through the entry-point group; `save_json(to_jsonable(batch))` and `load_json`
  round trip on a batch of arrays, paths and dataclasses.
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`tests/benchmarks/test_jsonable_benchmark.py`, the CI benchmark): the inputs of
  section 6 (flat `int` and `str`, dict, tree, dataclasses, scalar, small dict, deep list, array); the
  new code must not be slower than `main` on any of them except the array, where the gate is "at most
  6x `main` and at most 40 % of `json.dumps` of the result" (D12). A failing gate rejects the step.
- Coverage: 100 % of `src/coola/jsonable` (`--cov=coola`), no `# pragma: no cover`.

## 9. Decisions for the maintainers

**D1. The output is the JSON data model (tuples become lists).** _Recommended: yes._ The point of the
function is data that the standard library accepts and that survives `json.dumps` and `json.loads`
unchanged, which gives the idempotence property (section 1). `review/nested.md` kept tuples ("a tuple
stays a tuple") to be "structure-preserving", but a tuple is not JSON, a `deque` is not accepted, and
the sibling `recursive` property does not apply to a conversion whose purpose is to change the
representation. Alternative: keep tuples and `OrderedDict`; the caller then cannot tell whether the
result is JSON without calling `json.dumps`, and the property test of section 8 disappears.

Decision: follow recommendation

**D2. What happens to an unsupported type: three modes.** _Decided: `on_unsupported` with `"raise"`
(default), `"warn"` and `"keep"`._ The default is "strict and loud" (principle 3 of the architecture
guide): the error is raised at the leaf, with the type and the remedy. `"warn"` issues a
`UserWarning` per value and keeps it, so a caller migrating from `convert_to_jsonable` can see what
is not converted; `"keep"` keeps it silently (the behaviour of `main`, F1). Only the leaf function
differs between the modes (section 3), so there is one registry and no cost for `"keep"`. The price
is the one `isinstance` per leaf of the two checking modes (+0.03 µs) and the property "Accepted by
`json`", which holds in `"raise"` only. Rejected: a `default=` callable like `json.dumps` (the
registry already is the per-type hook); `repr` as a fourth mode (`ConvertTransformer(repr)` registered
for `object` in a copy of the registry does it).

**D3. Pydantic models: `model_dump()` in Python mode, then the registry.** _Recommended._ The registry
owns the semantics (principle 4): the sets are sorted, the arrays and the paths are converted, a
secret is masked, the field names, the exclusions (`Field(exclude=True)`) and the plain serializers of
the model are honoured by `model_dump`. Cost: a serializer declared `when_used="json"` (and the
`ser_json_*` settings) is not applied, because the dump is not made in JSON mode (the prototype gives
`{'v': 1}` where `model_dump(mode="json")` gives `{'v': 'v1'}`). Alternatives: `mode="json"` (today;
non-deterministic sets, fails on a NumPy field, F6); the fields as children, as `hashing` and
`summary` do (ignores `exclude` and every serializer). A model that needs its JSON mode registers
`ConvertTransformer(lambda m: m.model_dump(mode="json"))` for its class.

Decision: follow recommendation

**D4. `NaN` and infinities are kept.** _Recommended._ The contract is "what `json` accepts by default"
(as `save_json` documents), and `json.dumps(..., allow_nan=False)` is the one place that enforces
RFC 8259 and reports the offending value. Alternatives: an `allow_nan` keyword that raises (a second
check of what `json.dumps` already does), or `NaN` to `None` (silently changes a number into a
missing value, the kind of loss the properties forbid).

Decision: follow recommendation

**D5. Bytes are Base64.** _Recommended: keep._ It is the existing behaviour and the usual one for
binary data; it is lossless but indistinguishable from text on reading. Alternatives: raise (strict,
but the common case of a small blob then needs a registration), or UTF-8 decoding (raises for binary
data and returns text for text).

Decision: follow recommendation

**D6. A `timedelta` is a number of seconds.** _Recommended: keep._ Simple, no new code (`total_seconds`),
and the one that most consumers read. It is a `float`, so microseconds are lost above ~285 years.
Alternative: an ISO 8601 duration (`"P1DT0.000001S"`), lossless and what pydantic writes, but it needs
a formatter of about ten lines and a parser on the reading side.

Decision: follow recommendation

**D7. Frames and tables are not converted by default.** _Recommended._ A `DataFrame` has at least three
usual shapes (list of records, dict of columns, split) and none is neutral: a default would be a
silent choice for every caller. `pandas.Series` and `Index` are one-dimensional and are converted.
The guide shows the one-line registration for each shape. `polars`, `pyarrow`, `xarray` and `jax` are
the same decision (a jax array is `tolist()` and can be registered in one line, so it is a candidate
for the second round).

Decision: follow recommendation

**D8. A new package `coola.jsonable`, one function.** _Recommended._ The name avoids `coola.json`
(which hides the standard library in `from coola import json` and in the reader's head) and echoes
`to_jsonable`, the name that `pydantic_core` and FastAPI (`jsonable_encoder`) already teach.
`nested` keeps what it is about (shape of mappings and records). Alternative: keep it in `nested` and
only fix the code; cheaper, but `nested` then owns a registry, a plugin group and four library
adapters, and the guide of `nested` stays about two unrelated things.

Decision: follow recommendation

**D9. `register_transformers` and `ConvertTransformer` rather than a new function-based API.**
_Recommended._ The registry is the extension point of every package, the plugin group needs it, and
`ConvertTransformer(func)` makes the common case (a value to a JSON-able value) one line. A
`register_converters({Point: func})` would be a second spelling of the same thing.

Decision: follow recommendation

**D10. The keys go through the registry.** _Recommended._ It is why a `Path`, a `UUID`, an `Enum` or a
`datetime` key works, with no code in the mapping transformer, and a collision is reported once, at
the end. Alternative: only `str`, `int`, `float`, `bool` and `None` keys, anything else raises (the
rule of `json.dumps` itself); simpler, but `{Path("a"): 1}` then needs a pre-pass by the caller.

Decision: follow recommendation

**D11. Subclasses of the JSON scalars are kept.** _Recommended._ A `str` enum member, an `IntEnum`
member, a `numpy.str_` are `str` or `int` for the MRO, and `json` writes them as their base value.
The idempotence property holds with `==`. Normalising them would need a leaf function that checks the
exact type of every leaf of every call (an extra function call per leaf, and a rule per subclass) and
would not change what `json` writes. Alternative if a consumer needs exact types (YAML, a database
driver): register the enum class itself, `register_transformers({MyEnum: ConvertTransformer(lambda e:
e.value)})`, because the MRO then finds it before `str`.

Decision: follow recommendation

**D12. The output of an array is checked.** _Recommended._ The check is what makes "accepted by `json`"
true for complex numbers, object arrays and `datetime64` (F5), and it costs 0.09 µs per element
against 0.32 µs for `json.dumps` itself. Alternative if a user converts arrays of tens of millions of
elements: skip the check for the numeric dtypes (`kind in "biuf"`), a four-line `ArrayTransformer`
that returns `((), lambda _: data.tolist())`; not included until a benchmark of a real use asks for
it.

Decision: follow recommendation

**D13. No guard against a converter that never ends, other than its own type.** _Recommended._ The
engine has no depth limit by design ("Any depth"), so a transformer that returns a value of a type that
converts to a value of the first type (or a sequence whose elements are sequences of themselves,
`UserString`, F11) runs until memory is exhausted. `ConvertTransformer` rejects the one case it can see
(the same type); the contract test over the default registry catches the others; the guide states the
rule "a conversion must reduce the type to simpler ones". A depth counter in the engine would add a
check to every node for a mistake of a plugin author.

Decision: follow recommendation

## 10. Not covered

- **Reading back.** `from_jsonable`, a type tag (`{"__type__": "tuple"}`), a schema or a validation of
  the result: out of scope, the conversion is one way. Pydantic and `msgspec` do it with a schema.
- **Other formats** (YAML, TOML, MessagePack): the registry would be reusable (the structure is the
  same), but the lossy decisions (bytes, timedelta, dates) are different; a separate design.
- **Streaming and JSON Lines.** The output is built whole, like `recursive_apply`. A generator of
  records is not accepted (an iterator is a leaf, and a leaf of an unknown type raises).
- **Sorting the keys of the output**, indentation, ensure_ascii, `separators`: the job of
  `json.dumps` / `JsonSaver(**kwargs)`.
- **Frames, tables and datasets** (pandas `DataFrame`, polars, pyarrow, xarray): D7. **JAX arrays**: one
  line of registration, not included in the first round.
- **Faster dumping** (`orjson`, `ujson`): both accept the output of `to_jsonable`; they are not a
  dependency.
- **An `ensure_ascii`-like or locale-dependent text**, **big integers beyond 2^53** (a JSON reader in
  another language may lose them: the output keeps the Python `int`), **`NaN` handling** (D4).
- The tests of the other packages that exercise `convert_to_jsonable` only through `nested` (the
  records helpers) are not changed.

## Appendix: effect on the size of the code

| Module                                                      | Lines today | Expected after  | Why                                                                                                                  |
| ----------------------------------------------------------- | ----------- | --------------- | -------------------------------------------------------------------------------------------------------------------- |
| `nested/conversion.py` (JSON part)                          | 225         | 0               | moved                                                                                                                |
| `nested/conversion.py` -> `records.py`                      | 127         | 127             | the two record functions, unchanged                                                                                  |
| `jsonable/interface.py`                                     | 0           | about 140       | `to_jsonable`, the registry, `register_transformers`, the docstrings                                                 |
| `jsonable/transformers.py`                                  | 0           | about 150       | six transformers and `_to_dict`, `_sorted_elements`                                                                  |
| `jsonable/stdlib.py`                                        | 0           | about 60        | the table of the standard types                                                                                      |
| `jsonable/numpy.py`, `torch.py`, `pandas.py`, `pydantic.py` | 0           | about 25 each   | `get_default_transformers()` of one library                                                                          |
| tests (JSON part of `test_conversion.py`)                   | 48 tests    | about 150 tests | every transformer, every row of the table, every library, the properties (the target is the behaviour, not the size) |

The line counts are estimates from the prototype (about 120 lines of code for the containers, the
standard table and the libraries together, before docstrings) and from the code to delete; the gate
is the behaviour (sections 1 and 8), not the size. The source grows by about 225 lines, almost all
of them docstrings and the four library modules: the registry, the plugin group and the key handling
are new. What disappears is the table of predicates, the `sys.modules` look-ups, the registry built
per call and the unchecked outputs.
