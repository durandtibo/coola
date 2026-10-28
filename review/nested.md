# `coola.nested`: target state and the changes to reach it

This document describes the design that `coola.nested` should have, then the changes to make to the
current implementation (`main`, `8b53d01b`) to get there. The design follows the rules of the user
guides (`docs/docs/uguide/nested.md` **Properties**, `docs/docs/uguide/nested_data.md` **Properties
shared by all packages**) and the architecture guide (`docs/docs/dev/architecture.md`: registry
dispatch, MRO lookup, lazy imports, strict contracts). It stays aligned with the sibling packages
`equality`, `hashing`, `summary`, `iterator` and `recursive` (`review/recursive.md` is the model:
**one engine, one extension point, one rule per container type**). Its aim is a small package that is
easy to maintain: **no traversal of its own, no second definition of "container", no silent data
loss**. Breaking changes are accepted, and the future state is more important than the past one.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14, polars 2.0.0,
scripts kept outside the repository). Items marked **[verified]** were reproduced; **[reading]**
comes from the code. Nothing in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 1,543 lines (and 2,974 lines of tests) for 14
   functions. Four of them contain a hand-written recursion (`_flatten_into`, `_restore_lists`,
   `_rename_nested_keys`, and `to_jsonable._to_jsonable_recursive` that `convert_to_jsonable`
   depends on), a fifth builds a registry of ten types by hand. The target deletes these instead of
   adding code next to them: the traversal belongs to the engine of `recursive`, and `nested` only
   says what to do with a key or a container.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The release notes and the migration page list the changes.
3. **No slower than today.** The benchmark gate of section 8 must hold on the data sizes that matter
   (thousands of nodes). A fixed cost of a few microseconds per call on a tiny input is accepted and
   stated (section 6).
4. **Consistent with the siblings.** The same input must be handled the same way by `equality`,
   `hashing`, `summary`, `iterator`, `recursive` and `nested`: the same containers are entered
   (dataclasses included), iterators and text are leaves, cycles are detected, no `RecursionError`,
   options are keyword-only.
5. **Optional libraries stay lazy.** `import coola.nested` imports no optional library. The polars
   function is loaded on first use, the way `coola.summary` does it (`lazy_exports`).

## 1. Properties of the sub-package

`nested` is not one algorithm: it is a set of helpers built on the traversal packages. Its guide has
four vague properties today (non-destructive, type-agnostic, predictable, explicit failure). They are
replaced by the table below, which is also the content of the package docstring (as in
`coola.recursive`) and of the **Properties** section of the guide. Every choice of this document
exists to make one of them true **by construction** (in the engine, in one transformer) instead of by
the discipline of each function.

The functions fall into five groups. The properties say which group each one covers.

| Group      | Functions                                                                                            |
| ---------- | ---------------------------------------------------------------------------------------------------- |
| `flat`     | `to_flat_dict`, `from_flat_dict`, and `flatten_frame` (polars)                                       |
| `keys`     | `remove_keys_if`, `remove_keys_containing`, `remove_keys_starting_with`, `add_prefix_suffix_to_keys` |
| `mappings` | `merge_mappings`, `flatten_mapping`, `DuplicateKeyError`                                             |
| `records`  | `convert_to_dict_of_lists`, `convert_to_list_of_dicts`                                               |
| `json`     | `convert_to_jsonable`                                                                                |

| Property                | Statement                                                                                                                                                                                                                                                                                                                       | Groups                 | Enforced by                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------- |
| Pure                    | The input is never modified. A function that rebuilds containers returns new ones (the leaves and the values are shared, never copied). No function keeps state between calls.                                                                                                                                                  | all                    | the engine, stateless transformers            |
| One notion of container | A function that looks inside the values enters the containers of the other packages (`list`, `tuple`, `deque`, other sequences, mappings, sets, **dataclasses**) and treats text, bytes, iterators and every unknown type as a leaf. The only exception is documented: `to_flat_dict` does not enter sets (a set has no index). | `flat`, `keys`, `json` | the default registry of `recursive`           |
| Structure-preserving    | A function that rebuilds a structure keeps the type of every container, the top one included (`OrderedDict`, `defaultdict` with its factory, named tuples, subclasses, `deque` with its `maxlen`), and the order of the entries.                                                                                                | `keys`, `json`         | `coola.recursive.rebuild`                     |
| Any depth               | No `RecursionError`, whatever the nesting: the depth is bounded by memory only.                                                                                                                                                                                                                                                 | `flat`, `keys`, `json` | an explicit stack (the engine of `recursive`) |
| Cycle-safe              | A container that contains itself raises `ValueError`, never `RecursionError`.                                                                                                                                                                                                                                                   | `flat`, `keys`, `json` | the engine                                    |
| Keys are labels         | Only the `flat` and `keys` functions read or change keys. Elsewhere keys are kept as they are. `add_prefix_suffix_to_keys` renames the `str` keys and keeps the others; `to_flat_dict` writes every key with `str()`.                                                                                                           | `flat`, `keys`         | one key transformer                           |
| No silent loss          | Two entries are never merged into one without a decision of the caller: a key collision raises (`DuplicateKeyError` or `ValueError`), whatever the function (`to_flat_dict`, `from_flat_dict`, a rename, `merge_mappings`, `flatten_mapping`). A wrong input type raises `TypeError` naming the type.                           | all                    | one check per function, tested                |
| Deterministic           | The same input gives the same output in every process (whatever `PYTHONHASHSEED` is): the entries keep the order of the input; a set becomes a sorted list, and the sort never depends on the iteration order of the set.                                                                                                       | all                    | `convert_to_jsonable`, `merge_mappings`       |
| Documented round trips  | `from_flat_dict(to_flat_dict(x)) == x` for `dict`/`str` keys that do not contain the separator (or with `escape=True`); with `restore_lists=True`, also for lists. `convert_to_list_of_dicts(convert_to_dict_of_lists(r)) == r` for non-empty records.                                                                          | `flat`, `records`      | property tests (`tests/properties`)           |
| Explicit options        | Options are keyword-only. An invalid option raises `ValueError` before any work is done, and names the valid values.                                                                                                                                                                                                            | all                    | the signatures                                |
| Lazy                    | `import coola.nested` imports no optional library (polars is loaded by the first use of `flatten_frame`).                                                                                                                                                                                                                       | all                    | `lazy_exports`                                |
| Thread-safe             | The state of a call is local to the call. A registry is read, never modified; there is no module-level mutable state.                                                                                                                                                                                                           | all                    | stateless functions                           |
| Fast                    | The cost is linear in the number of nodes.                                                                                                                                                                                                                                                                                      | all                    | the engine                                    |

Not guaranteed, and documented as such:

- **Shared references.** A container reachable by several paths is visited once per path (the rule
  of `recursive` and `iterator`), so `to_flat_dict` of `[x, x]` has the entries of `x` twice.
- **Dataclasses do not come back.** `to_flat_dict` enters a dataclass and writes its fields;
  `from_flat_dict` gives dictionaries. The round trip is for dicts and lists only.
- **The type of a list-like.** `to_flat_dict` gives the same entries for a list and a tuple, so
  `from_flat_dict(..., restore_lists=True)` returns lists.
- **Non-`str` keys of `to_flat_dict`.** They are written with `str()`: `(1, 2)` becomes `"(1, 2)"`
  and cannot be restored.
- **Strict JSON.** `convert_to_jsonable` follows the Python `json` module: `NaN` is kept, the keys are
  not converted, tuples stay tuples.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Status         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **Four hand-written recursions.** `to_flat_dict` (`_flatten_into`, eight keyword parameters, its own `_check_no_cycle`), `from_flat_dict` (`_restore_lists`), `add_prefix_suffix_to_keys(recursive=True)` (`_rename_nested_keys`) and `to_jsonable` (`_to_jsonable_recursive`, used for the fields of a dataclass) recurse in Python. A structure nested 5,000 levels raises `RecursionError` in the first three, `restore_lists` on 3,000 levels too, and `convert_to_jsonable` on 2,000 nested dataclasses. The `equality`, `recursive` and `iterator` engines have no such limit.                                                                                      | **[verified]** |
| F2  | **A cycle is a `RecursionError` in `add_prefix_suffix_to_keys`**, not the `ValueError` of the shared property (`d["s"] = d`). `to_flat_dict` has its own cycle check (a third implementation after the engines of `recursive` and `iterator`).                                                                                                                                                                                                                                                                                                                                                                                                                            | **[verified]** |
| F3  | **Five definitions of "container".** `to_flat_dict`: `Mapping` and non-text `Sequence` (a `range` and a named tuple are entered), sets and dataclasses are leaves. `add_prefix_suffix_to_keys`: `Mapping` and `Sequence`; sets and dataclasses are leaves. `remove_keys_if`: a registry of ten types written by hand in `_make_key_filter_registry` (no `MappingProxyType`, dataclasses are leaves, pinned by `test_remove_keys_if_dataclass_is_a_leaf`). `convert_to_jsonable`: the default registry with the dataclass fallback replaced by a leaf. `recursive_apply`: dataclasses are entered. `nested_data.md` says that a dataclass is a container in every package. | **[verified]** |
| F4  | **The type is lost, and the annotation lies.** `add_prefix_suffix_to_keys({"p": OrderedDict(a=1)}, "x_", recursive=True)` gives a plain `dict` at depth 1 but rebuilds a named tuple in a list. `remove_keys_if` is documented and annotated `-> dict` and returns an `OrderedDict`, a `defaultdict`, a flat `ChainMap`, or even the list for `remove_keys_if([1], ...)` (no check on the type of the input).                                                                                                                                                                                                                                                             | **[verified]** |
| F5  | **`convert_to_jsonable` is not deterministic.** A set in a dataclass is converted by `to_jsonable` itself, in iteration order: `['d','a','b','c','e']` with `PYTHONHASHSEED=1`, `['c','b','d','a','e']` with 2 (outside a dataclass the same set is sorted). A tuple in a dataclass becomes a list, a tuple outside stays a tuple. `try_sorted_list` falls back to the iteration order of the set for mixed types (`{1, "a"}`), and sorting values with a `NaN` depends on the input order.                                                                                                                                                                               | **[verified]** |
| F6  | **`KeyFilterTransformer` is in the wrong package.** It lives in `coola.recursive` (and contradicts its property "keys are never transformed") but its only user is `nested.mapping`. The registry that `remove_keys_if` builds per call copies the default one by hand and drifts from it (F3).                                                                                                                                                                                                                                                                                                                                                                           | **[reading]**  |
| F7  | **Positional options.** `to_flat_dict(data, prefix, separator, to_str)`, `flatten_mapping(mapping, on_duplicate, always_prefix, separator)`, `merge_mappings(mappings, on_duplicate)`, `add_prefix_suffix_to_keys(mapping, prefix, suffix, recursive)` accept their options by position (`to_flat_dict({"a": 1}, "p", "/")` works). The `CLAUDE.md` rule is keyword-only.                                                                                                                                                                                                                                                                                                 | **[verified]** |
| F8  | **Public surface without a reason.** `get_first_value` has no user in `src` and is `next(iter(d.values()))` plus an error message. `try_sorted_list` has one user (`convert_to_jsonable`). `SetToListTransformer` is importable (public name, not in `__all__`). Four of the five names of `coola.nested.polars` are implementation details (`is_nested_struct`, `unnest_with_separator`, `unnest_one_level`, `expand_list_columns`) and none of the module is documented.                                                                                                                                                                                                | **[reading]**  |
| F9  | **`convert_to_list_of_dicts` accepts a set** and pairs the elements of several sets in an arbitrary order (`{"a": {1, 2}}` gives `[{'a': 1}, {'a': 2}]`). A generator gives a raw `TypeError: object of type 'generator' has no len()`, `convert_to_dict_of_lists` on an iterator gives `'list_iterator' object is not subscriptable`.                                                                                                                                                                                                                                                                                                                                    | **[verified]** |
| F10 | **`from_flat_dict`.** Annotated `dict[str, Any]` but any mapping works. `restore_lists=True` also converts a `dict` given by the caller as a value (`{"a": {"0": 1}}` gives `{"a": [1]}`), although the function carries a set of ids (`created`) to avoid touching the dicts of the caller during the build.                                                                                                                                                                                                                                                                                                                                                             | **[verified]** |
| F11 | **`merge_mappings(on_duplicate="suffix")` depends on the order of the mappings**: `[{"a":1},{"a":2},{"a_1":3}]` raises `DuplicateKeyError`, `[{"a_1":3},{"a":1},{"a":2}]` gives `{'a_1': 3, 'a': 1, 'a_2': 2}`. `flatten_mapping(on_duplicate="prefix")` moves the renamed first occurrence to the end of the output (`{'acc': 2, 'train.loss': 1, ...}` for an input that starts with `loss`).                                                                                                                                                                                                                                                                           | **[verified]** |
| F12 | **`flatten_frame` (polars).** Accepts a `LazyFrame` although it is annotated and documented for a `DataFrame` (it resolves the schema four times and emits a `PerformanceWarning`). The version compatibility is feature detection by `except TypeError` (`unnest_with_separator`, `_explode`), which also hides an unrelated `TypeError`. Not exported by `coola.nested`, not in the guide.                                                                                                                                                                                                                                                                              | **[verified]** |
| F13 | **The guide is thin and partly wrong.** The four properties are generic; "Type-agnostic: dict, list, tuple" is not what the code does (F3); `to_flat_dict` is filed under "Mapping Utilities"; `flatten_frame` is absent; "Design Principles" repeats "Properties"; the package docstring is one line (the other packages list their properties there).                                                                                                                                                                                                                                                                                                                   | **[reading]**  |

What is already right and must not regress: the `escape` option and the conflict errors of
`to_flat_dict`/`from_flat_dict` (a conflict raises, a dict given by the caller is never mutated); the
empty containers kept as values of the flat dict; `DuplicateKeyError(KeyError, ValueError)` with a
readable message; the comparison of duplicate values with `objects_are_equal(equal_nan=...)`; the
validation of `on_duplicate` before any work; the explicit rejection of `str`/`bytes` in
`convert_to_list_of_dicts`; the lazy import of polars.

## 3. Architecture

```
coola.nested
  flat.py        to_flat_dict, from_flat_dict            one explicit-stack loop + one transformer (_Node)
  keys.py        remove_keys_*, add_prefix_suffix_*      one private transformer (_KeyTransformer) + recursive_apply
  mappings.py    merge_mappings, flatten_mapping         plain loops over the top-level mappings, no nesting
  records.py     convert_to_dict_of_lists, ..._of_dicts  plain loops, no nesting
  json.py        convert_to_jsonable                     recursive_apply with two transformers (dataclass, set)
  polars.py      flatten_frame (+ private helpers)       loaded on first use
```

The rule: **a function that goes inside values uses the engine of `recursive` and describes one
type; a function that works on the top level only is a plain loop.** `merge_mappings`,
`flatten_mapping`, `convert_to_dict_of_lists` and `convert_to_list_of_dicts` do not look inside the
values; they are right as they are, apart from the points of section 4. The modules are renamed to
the plural of their content (`mapping.py` becomes `mappings.py`, `conversion.py` is split) only if the
split of step 7 is done; the names of the functions do not depend on it.

### 3.1 `keys`: one private transformer (F1, F2, F3, F4, F6)

`KeyFilterTransformer` moves from `coola.recursive` to `coola.nested.keys`, becomes private, and
learns to rename. It is the only code that changes keys.

```python
class _KeyTransformer(BaseTransformer[Mapping[Any, Any]]):
    def __init__(self, *, rename=None, drop=None):
        self._rename, self._drop = rename, drop

    def expand(self, data):
        keys = [key for key in data if not (self._drop and self._drop(key))]
        new = [self._rename(key) for key in keys] if self._rename else keys
        if len(set(new)) != len(new):  # no silent loss
            raise DuplicateKeyError(
                f"Renaming the keys of a {type(data).__name__} gives duplicate keys"
            )
        return [data[key] for key in keys], lambda items: rebuild_mapping(
            data, zip(new, items, strict=True)
        )


def _apply(data, **options):
    registry = get_default_registry().copy()
    transformer = _KeyTransformer(**options)
    for data_type in (dict, Mapping, MappingProxyType):  # the mapping types of the default registry
        registry.register(data_type, transformer, exist_ok=True)
    return recursive_apply(data, _identity, registry=registry)
```

- `remove_keys_if(mapping, predicate)` is `_apply(mapping, drop=predicate)`;
  `remove_keys_containing` and `remove_keys_starting_with` stay thin wrappers (a predicate on `str`
  keys; a non-`str` key is kept).
- `add_prefix_suffix_to_keys(mapping, prefix="", suffix="", *, recursive=False)` is
  `_apply(mapping, rename=...)` when `recursive`, and one `rebuild_mapping` call on the top level
  otherwise (no engine needed). A renamed key that collides with another key raises `DuplicateKeyError`
  (impossible today for a prefix and a suffix because the map is injective on `str`, but the
  transformer is generic and a rename function is one step away).
- The registry is **derived from the default one**, so the containers are the ones of every other
  package (F3): dataclasses are entered (the fields that are mappings are filtered or renamed, the
  dataclass is rebuilt), `MappingProxyType`, `deque` and the user-registered containers follow. The
  list of ten types is deleted.
- The function of the values is the identity. A leaf costs one call of `_identity`, which the engine
  makes in about 0.1 µs; the prototype below shows no measurable cost on trees.
- The top-level type is kept (an `OrderedDict` gives an `OrderedDict`) and the annotation is a
  `TypeVar` bound to `Mapping`, so `-> dict` stops lying (F4). A non-mapping input is a `TypeError`
  naming the type (today `remove_keys_if([1], ...)` returns the list).
- `KeyFilterTransformer` is deleted from `coola.recursive` (`__all__`, guide, tests): `recursive`
  recovers the property "keys are never transformed" without an exception, and `review/recursive.md`
  section 10 ("`MapKeysTransformer` is the natural next transformer") is answered here, in the one
  package that needs it.

### 3.2 `flat`

**`to_flat_dict(data, *, prefix=None, separator=".", to_str=None, escape=False)`.** One loop on an
explicit stack replaces `_flatten_into`, `_check_no_cycle`, `_build_key` and the recursion.

- The labels of the children are the only knowledge `nested` must have of the containers, because the
  traversal packages return the children without their label (`iterator` plans `dfs_items` for that,
  `review/iterator.md` D9). They are given by one function `_items(data)` of about ten lines:
  `Mapping` gives `(str(key), value)` (the escape is applied to the label), a non-text `Sequence`
  gives `(str(i), item)`, a dataclass gives `(field name, value)`, everything else is a leaf. Sets are
  leaves (documented exception: no index); `bytes`, `bytearray`, `str` and iterators are leaves.
  Dataclasses are entered, like in every package (behaviour change, F3).
- **When `dfs_items` exists** (iterator step 8), `_items` and the loop are deleted and
  `to_flat_dict` is the join of the paths; the only thing it keeps is the empty containers (see the
  next point). Do not wait for it: the loop is 25 lines.
- An empty container is kept as the value of its key (`{"a": {}}` gives `{"a": {}}`), as today, so a
  round trip does not lose it. The root being empty gives `{}`.
- A cycle raises `ValueError` through the stack of the ancestors of the loop (the shared message of
  the other engines: "Cycle detected: an object of type ... contains itself").
- A key collision raises `ValueError` (kept). `to_str` is kept (a leaf type stringified with `str`),
  `prefix`, `separator`, `to_str`, `escape` are keyword-only (F7).
- `prefix=""` is a prefix like any other (`".a"`). Documented; no special case.

**`from_flat_dict(data, *, separator=".", restore_lists=False, escape=False)`.** The build loop
(`_set_nested`, `_split_key`) stays: it is flat, not recursive. Two changes remove the id set and the
recursion:

- The nodes created by the function are instances of a private `dict` subclass, `_Node`. A value
  given by the caller, even a `dict`, is a plain `dict`, so `isinstance(existing, _Node)` is the test
  "this dict is mine" (it replaces the set `created` of ids) and a dict of the caller is **never
  converted by `restore_lists`** (F10).
- The result is built by a final `recursive_apply` with a registry derived from the default one in
  which `_Node` has a transformer: it rebuilds a plain `dict`, or a `list` when `restore_lists` is set
  and the keys are exactly `"0"`, ..., `"n-1"`. The root is a separate subclass whose transformer
  never builds a list (the root is always a `dict`). `_restore_lists` is deleted, and the depth is no
  longer limited (F1). The prototype of section 6 gives the same output as `main` on the benchmark
  tree and runs on 20,000 levels.
- `data` is annotated `Mapping[str, Any]` (F10).

### 3.3 `json`: `convert_to_jsonable` and `to_jsonable` (F1, F3, F5)

`convert_to_jsonable(data)` is `recursive_apply(data, to_jsonable, registry=registry)` with a registry
derived from the default one in which:

- `set` and `frozenset` have a transformer whose `rebuild` returns a sorted list (kept from
  `SetToListTransformer`, now private), sorted with a key that does not depend on the iteration order:
  the elements when they compare, otherwise `(type name, repr)`. `try_sorted_list` becomes the private
  `_sorted_elements` of this module (F8);
- a dataclass has a transformer that returns its fields as a `dict` (`expand` gives the field values,
  `rebuild` zips them with the names). The dataclass fallback of the default registry (entering and
  rebuilding the same dataclass) is replaced, so **a dataclass is converted by the engine, field by
  field, like any other container** and its fields get the same treatment as everything else: sets
  are sorted, the depth is unlimited, a cycle is a `ValueError`.

`coola.utils.conversion.to_jsonable` converts **one object** and no longer walks the fields of a
dataclass: the `is_dataclass` branch and `_to_jsonable_recursive` are deleted (about 30 lines and a
second recursion). Its docstring says "use `convert_to_jsonable` for a dataclass". It is a breaking
change for `to_jsonable(Point(1, 2))` (the object is returned unchanged). The behaviour changes of
`convert_to_jsonable`: a tuple in a dataclass stays a tuple (before: list), a set in a dataclass is
sorted (before: iteration order).

The registry is derived at each call (it reads the user registrations of the default registry; a copy
costs 6 µs). It is not cached in a `LazySingleton`, which would freeze the default registry at the
first call.

`convert_to_jsonable` documents what it does not do (keys, `NaN`, tuples) as today. A
`stringify_keys` option is the obvious next step and is not part of this design (section 10).

### 3.4 `mappings` and `records`: small corrections

No new mechanism. These functions work on the top level only.

- **Options keyword-only** (F7): `merge_mappings(mappings, *, on_duplicate="raise", equal_nan=False)`,
  `flatten_mapping(mapping, *, on_duplicate="raise", always_prefix=False, separator=".",
equal_nan=False)`.
- **`merge_mappings(on_duplicate="suffix")` does not depend on the order** (F11). A key of an input
  mapping that equals a generated suffixed key is a conflict whatever the position: the check is made
  once, on the set of all input keys, before the merge (a `set` of the keys of every mapping, then
  `_free_suffixed_key` skips them). The result for `[{"a_1":3},{"a":1},{"a":2}]` is
  `{'a_1': 3, 'a': 1, 'a_2': 2}` and the result for the other order is the same entries (the order of
  the keys follows the input). This removes the `raise` of "it collides with a suffixed key": a
  generated key never overwrites or hides an input key, so there is nothing to raise. The error of the
  docstring (`DuplicateKeyError` "if a key of an input mapping is equal to a key generated by the
  suffix strategy") goes away. The unit tests of the old error become tests of the new result.
- **`flatten_mapping(on_duplicate="prefix")` keeps the order of the input** (F11): the first
  occurrence is renamed in place (the dict is rebuilt at the end instead of `pop` then re-adding), so
  `{'train.loss': 1, 'acc': 2, 'val.loss': 3}` for the input of the guide. Both functions can then
  say "the output follows the order of the input". Behaviour change, documented.
- **`convert_to_list_of_dicts`** rejects `Set` values with a `TypeError` naming the type, like it
  already does for text (F9), and accepts any `Sequence` (a generator is not one: `TypeError` that
  names the type instead of the `len` error). **`convert_to_dict_of_lists`** accepts any iterable of
  mappings (it materialises it once), so a generator works.
- **`get_first_value` is deleted** (F8; decision D1). `next(iter(data.values()))` is the idiom; the
  error message was the only value.

### 3.5 `polars`: `flatten_frame` (F8, F12)

- `flatten_frame(frame, *, separator=".", depth=None, expand_lists=False)` is exported lazily from
  `coola.nested` with `lazy_exports` (`from coola.nested import flatten_frame` works without
  importing polars at `import coola.nested`), documented in the guide with the example of its
  docstring, and listed in `tests/integration/test_lazy_imports.py`.
- The four helpers become private (`_is_nested_struct`, `_unnest_with_separator`, `_unnest_one_level`,
  `_expand_list_columns`); the tests of their behaviour go through `flatten_frame`, which keeps the
  coverage.
- Only a `DataFrame` is accepted: a `LazyFrame` raises a `TypeError` that names the type and says
  "call `collect()`" (a lazy frame has no cheap schema, and the row count of an exploded list is only
  known after the evaluation). It is a visible rule instead of a warning at the end of a long chain.
- The version checks use `coola.utils.version.compare_version` (feature gate on the version where
  `unnest(separator=...)` and `explode(empty_as_null=...)` appeared) instead of `except TypeError`, so
  a real `TypeError` propagates. If the minimum supported polars version rises, the two branches are
  deleted (decision D3).

## 4. Public API after the change

```python
from coola.nested import (
    DuplicateKeyError,
    add_prefix_suffix_to_keys,  # (mapping, prefix="", suffix="", *, recursive=False)
    convert_to_dict_of_lists,  # (records)
    convert_to_jsonable,  # (data)
    convert_to_list_of_dicts,  # (columns)
    flatten_frame,  # (frame, *, separator=".", depth=None, expand_lists=False), lazy, polars
    flatten_mapping,  # (mapping, *, on_duplicate="raise", always_prefix=False, separator=".", equal_nan=False)
    from_flat_dict,  # (data, *, separator=".", restore_lists=False, escape=False)
    merge_mappings,  # (mappings, *, on_duplicate="raise", equal_nan=False)
    remove_keys_containing,  # (mapping, substring)
    remove_keys_if,  # (mapping, predicate)
    remove_keys_starting_with,  # (mapping, prefix)
    to_flat_dict,  # (data, *, prefix=None, separator=".", to_str=None, escape=False)
)
```

Removed: `get_first_value`, `try_sorted_list` (private), `SetToListTransformer` (private),
`coola.nested.polars.{is_nested_struct, unnest_with_separator, unnest_one_level, expand_list_columns}`
(private), `coola.recursive.KeyFilterTransformer` (moved, private), the positional options, the
dataclass branch of `coola.utils.conversion.to_jsonable`. The names of the modules (`flat`, `mapping`,
`conversion`, `polars`) are not public contracts (the package root is) and can be kept to keep the
diff of each step small; the split of section 3 is optional.

## 5. Dependents

| Package                                         | Change                                                                                                                                                                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.recursive`                               | `KeyFilterTransformer` and `key.py` deleted (guide, `__all__`, tests). `rebuild_mapping` and `rebuild_sequence` stay the public helpers (`nested.keys` imports `rebuild_mapping`). The **Properties** of `recursive` lose the implicit exception. |
| `coola.utils.conversion.to_jsonable`            | The dataclass branch and `_to_jsonable_recursive` are deleted; the docstring points to `convert_to_jsonable`.                                                                                                                                     |
| `coola.io.json` (`save_json`)                   | The docstring already points to `convert_to_jsonable`; no code change.                                                                                                                                                                            |
| `tests/integration/recursive/test_dataclass.py` | `test_remove_keys_if_dataclass_is_a_leaf` is replaced by `test_remove_keys_if_enters_the_dataclass`; `test_convert_to_jsonable_dataclass_is_converted_whole` stays and gains the set and the tuple of F5.                                         |
| `tests/package_checks.py`                       | Unchanged (`coola.nested.to_flat_dict`).                                                                                                                                                                                                          |
| `iterator`                                      | None now. `dfs_items` (D9 of `review/iterator.md`) later replaces `_items` of `to_flat_dict`.                                                                                                                                                     |
| Docs                                            | `uguide/nested.md` rewritten (section 7, step 9), `uguide/nested_data.md` (row of `nested` in the table), `dev/architecture.md` (one paragraph), migration page, `refs/nested.md` (the polars function appears through the lazy export).          |

## 6. Evidence: a prototype

A prototype of `_KeyTransformer`, of the `_Node` build of `from_flat_dict` and of the two transformers
of `convert_to_jsonable` was run on `main` on top of the real engine of `recursive` (default registry
copy, real `rebuild_mapping`). It is not part of the repository. **[verified]**

| Check                                                                       | `main`                             | Prototype                       |
| --------------------------------------------------------------------------- | ---------------------------------- | ------------------------------- |
| `remove_keys_if` on a tree of 7,776 leaves                                  | 5.5 to 5.9 ms                      | 2.9 to 3.6 ms                   |
| `add_prefix_suffix_to_keys(recursive=True)` on the same tree                | 3.0 to 3.4 ms                      | 3.7 to 4.2 ms                   |
| `from_flat_dict` on the 7,776 flat entries (same output as `main`)          | 7.7 to 7.9 ms                      | 7.2 to 8.0 ms                   |
| `remove_keys_if` on `{"a": 1, "b": {"c": 2}}`                               | 5.2 to 5.8 µs                      | 12 to 13.5 µs                   |
| rename on 20,000 levels                                                     | `RecursionError`                   | works (180,001 chars)           |
| `from_flat_dict(restore_lists=True)` with 20,000 levels                     | `RecursionError` (about 3,000)     | works                           |
| rename of a cyclic dict                                                     | `RecursionError`                   | `ValueError`                    |
| `OrderedDict` and a named type through `remove_keys_if`                     | `OrderedDict` kept at the top only | kept at every depth             |
| dataclass with a dict field through `remove_keys_if`                        | kept as a leaf                     | `{'x': Node('n', [{'k2': 2}])}` |
| `convert_to_jsonable({"p": P({"b","a","c"}, (1,2))})`, `PYTHONHASHSEED=1,2` | different lists                    | `['a', 'b', 'c']` both          |

Reading the numbers: the engine is faster on large trees (removal 1.5 to 2 times) and the same
elsewhere, and a call costs about 7 µs more on a tiny input, which is the copy of the default registry
(6.4 µs measured) and the registrations. Two mitigations are available if the benchmark gate rejects
it: build the derived registry lazily once per `(function, options)` through a `LazySingleton` keyed
on the registry version (not worth the complexity today), or pass `recursive_apply` a ready transformer
override. The gate of section 8 states the accepted budget.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                          | Breaking                             |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| 1    | **Properties first (F13).** Package docstring, **Properties** table of section 1 in the guide, remove "Design Principles", file `to_flat_dict` under its own heading, add `flatten_frame`. Documentation only; statements that the code does not honour yet carry "(from step N)" until the step lands, or the PR is merged after the steps.    | no                                   |
| 2    | **Keyword-only options and dead API (F7, F8).** Options of the five functions keyword-only, `get_first_value` deleted, `try_sorted_list` and `SetToListTransformer` private, the four polars helpers private, migration notes.                                                                                                                  | yes                                  |
| 3    | **Keys on the engine (F1, F2, F3, F4, F6).** `_KeyTransformer`, `remove_keys_*` and `add_prefix_suffix_to_keys` on `recursive_apply`, type kept, `TypeError` for a non-mapping, `DuplicateKeyError` on a collision, `KeyFilterTransformer` deleted from `recursive`. Regression tests: cycle, 20,000 levels, `OrderedDict` at depth, dataclass. | yes (types, dataclass)               |
| 4    | **`from_flat_dict` without recursion (F1, F10).** `_Node`, final `recursive_apply`, the `created` set deleted, a caller dict is never converted, `Mapping` annotation. Regression test: 20,000 levels, a dict value kept with `restore_lists`.                                                                                                  | yes (value dicts of the caller kept) |
| 5    | **`convert_to_jsonable` deterministic (F5).** Dataclass and set transformers, `_sorted_elements`, `to_jsonable` loses the dataclass branch. Regression tests with several `PYTHONHASHSEED` values in `tests/integration/` (a subprocess is the only way to change it).                                                                          | yes                                  |
| 6    | **`to_flat_dict` on an explicit stack (F1, F2, F3).** One loop and `_items`; dataclasses entered; cycle `ValueError`; keyword-only options (can be merged into step 2). Regression tests: 20,000 levels, a dataclass, a cycle.                                                                                                                  | yes (dataclass entries)              |
| 7    | **Mappings and records (F9, F11).** `merge_mappings` suffix order-independent, `flatten_mapping` prefix keeps the order, sets rejected by `convert_to_list_of_dicts`, `convert_to_dict_of_lists` takes any iterable. Optional: the split of the modules.                                                                                        | yes (outputs)                        |
| 8    | **polars (F12).** `flatten_frame` exported lazily and documented, `DataFrame` only, version checks by `compare_version`.                                                                                                                                                                                                                        | yes (`LazyFrame`)                    |
| 9    | **Documentation.** Rewrite `uguide/nested.md`, the `nested` row of `nested_data.md`, the paragraph of `dev/architecture.md`, the migration page, and the status table of `review/suggestions.md` (items 25.3, 26.2 with the PR numbers).                                                                                                        | no                                   |

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/nested/`); the conventions of `CLAUDE.md` apply (one
behaviour per test, literal expected values, a regression test for every bug checked with
`git stash push src`, no local imports, the `*_available` fixtures for polars).

- **Keys** (`test_keys.py`): the predicate at every depth, in lists, tuples, sets and dataclasses; the
  whole output of a mixed structure against a literal; `OrderedDict`, `defaultdict` (factory kept),
  named tuple, `MappingProxyType` kept; the top-level type kept; a non-mapping raises `TypeError`; a
  rename collision raises `DuplicateKeyError`; a cycle raises `ValueError`; 20,000 levels do not
  raise; the input is not modified and shares no container with the output; `recursive=False` renames
  the top level only; non-`str` keys are kept.
- **Flat** (`test_flat.py`): the 686 existing lines stay; add the dataclass, the `range`, the cycle,
  the 20,000 levels, the shared container (entries twice), the set kept as a leaf, the empty
  containers; `from_flat_dict`: a value `dict` is not converted by `restore_lists`, 20,000 levels, the
  root is always a dict, the `Mapping` input.
- **JSON** (`test_conversion.py`): the dataclass with a set and a tuple (whole literal), a set of
  mixed types, the depth and the cycle through a dataclass, an Enum and a path inside a dataclass, the
  determinism test over several `PYTHONHASHSEED` values in `tests/integration/nested/`.
- **Mappings and records**: order independence of `suffix` (parametrised over the permutations of a
  three-mapping case), the order of `prefix`, a set value rejected, a generator accepted by
  `convert_to_dict_of_lists`.
- **Polars**: the existing 853 lines are kept through the public function; the `LazyFrame` error; the
  lazy import (`import coola.nested` does not import polars: `tests/integration/test_lazy_imports.py`).
- **Properties** (`tests/properties/test_nested.py`, extend): the round trips of section 1 (kept),
  **the cycle and depth properties** for `to_flat_dict`, `remove_keys_if` and `convert_to_jsonable`
  on the strategies of `tests/properties/strategies.py` (new), the type of the containers is kept by
  `remove_keys_if` with a predicate that never matches (new: it is the identity), the input is not
  modified (new).
- **Integration** (`tests/integration/nested/`): the dataclass behaviour with the real default registry
  (no patching), a custom container registered with `register_transformers` that `remove_keys_if`
  enters, and a thread test with `run_threads` on `remove_keys_if` and `convert_to_jsonable` (a lock
  around the shared list, docstring "Test that ...").
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`tests/benchmarks/test_nested_benchmark.py`, new; the package has none today):
  `to_flat_dict`, `from_flat_dict`, `remove_keys_if`, `add_prefix_suffix_to_keys(recursive=True)` and
  `convert_to_jsonable` on a tree of about 8,000 leaves and on a flat dict of 100,000 entries, plus a
  tiny input per function. A step is rejected if the large inputs are more than 10 % slower than
  `main` or a tiny input costs more than 15 µs.
- Coverage: 100 % of `src/coola/nested` (`--cov=coola`), no `# pragma: no cover` (the `else:  # pragma:
no cover` of the polars fallback import is the existing exception for an absent library).

## 9. Decisions for the maintainers

**D1. Delete `get_first_value`?** _Recommended: yes._ It is a one-liner without user in `src`, and the
package shrinks to functions that do real work. The cost is one line for the callers
(`next(iter(data.values()))`, which raises `StopIteration` on an empty mapping). Alternative: keep it;
it costs nothing to maintain, but it is the only function of the package that nobody needs from a
nested structure.

Decision: move it to coola.utils.mapping

**D2. Own the key transformer in `nested`, or generalise it in `recursive`?** _Recommended: in
`nested`, private._ `recursive` says "keys are never transformed", and a transformer that renames or
drops keys is the exception that only `nested` needs. Alternative: a public `MapKeysTransformer` in
`recursive`, which `review/recursive.md` section 10 foresees. It becomes the right home as soon as a
second package needs it; moving a private class is a rename.

Decision: follow recommendation

**D3. Raise the minimum polars version?** _Recommended: look at the support policy of
`pyproject.toml` (`polars>=1.0,<3.0`) when step 8 is made._ If the oldest supported version has
`unnest(separator=...)` and `explode(empty_as_null=...)`, the two compatibility branches
(`unnest_with_separator`, `_explode`) are deleted, about 25 lines and two untestable branches on
current CI. If not, keep them behind `compare_version`.

Decision: keep `polars>=1.0,<3.0`

**D4. Enter dataclasses in `to_flat_dict`, `remove_keys_*` and `add_prefix_suffix_to_keys`?**
_Recommended: yes._ `nested_data.md` says that a dataclass is a container in every package and
`recursive_apply` and `convert_to_jsonable` enter them already; five different rules for the same
object (F3) is the real cost. The price is a behaviour change for code that relied on a dataclass
being a leaf of the flat dict, and that a dataclass does not come back from `from_flat_dict`.
Alternative: keep dataclasses as leaves of `nested` and document it; the table of
`nested_data.md` then gets an exception per function.

Decision: follow recommendation

**D5. Keep `recursive=False` as the default of `add_prefix_suffix_to_keys`?** _Recommended: yes, as a
keyword-only option._ The common use (a prefix such as `train/` on the top-level metrics) is
non-recursive, and the `remove_keys_*` functions are always recursive because removing a key at the
top only is a dict comprehension. Alternative: make the rename always recursive and delete the flag
(one branch less); users of the top level write `{f"x_{k}": v for ...}`.

Decision: follow recommendation

**D6. Should a collision on rename raise, or should the last entry win?** _Recommended: raise_
(`DuplicateKeyError`, the exception of the package). It is the rule for every other function that can
merge two entries and the only one that makes a rename function safe to expose later.

Decision: follow recommendation

**D7. Cache the derived registries?** _Recommended: no._ A copy of the default registry is 6 µs, which
is below the noise of any call on a real structure, and a cache of derived registries must track the
changes of the default registry (a user registers a transformer after the first call). Revisit if the
benchmark of tiny inputs matters to a user.

Decision: follow recommendation

**D8. `to_flat_dict` on top of `dfs_items` or its own loop?** _Recommended: its own loop now
(25 lines, section 3.2), `dfs_items` later._ The path-aware iterator is not built, the loop is small
and fixes the depth, the cycle and the dataclass today, and deleting it when `dfs_items` lands is a
local change. Alternative: make `dfs_items` first and keep `to_flat_dict` as is; it blocks the fix on
an unrelated feature.

Decision: make `dfs_items` first

## 10. Not covered

- **Keys of `convert_to_jsonable`.** A `stringify_keys` option (or a `str`/`int`/`float`/`bool`/`None`
  check that raises) would make the output strictly accepted by `json.dumps`. Natural next step with
  `_KeyTransformer`; it is not part of this design.
- **Strict JSON** (`NaN`, infinities): the contract stays "accepted by the Python `json` module".
- **Dataclasses and pydantic models from `from_flat_dict`**: the function builds dicts and lists;
  rebuilding a typed object needs a schema and is not a flat-dict operation.
- **Lists of dicts to a flat table and back** (`pandas`/`polars` do it): `records` stays two
  functions of the standard library.
- **An element-wise `rename_keys(data, func)` function**: `_KeyTransformer` supports it in one line;
  the public function is added when a user needs a rename other than a prefix and a suffix.
- **`merge_mappings` of nested mappings (deep merge).** Not a function of the package; the top-level
  merge with a duplicate strategy is what exists.
- **Parallel or lazy application**: the outputs are built whole, as in `recursive`.

## Appendix: effect on the size of the package

| Module                | Lines today | Expected after | Why                                                                                                                               |
| --------------------- | ----------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `flat.py`             | 456         | about 330      | one loop and `_items` instead of `_flatten_into`, `_check_no_cycle`, `_build_key`; `_Node` instead of `created`, `_restore_lists` |
| `mapping.py`          | 562         | about 450      | `get_first_value`, `_rename_nested_keys`, `_make_key_filter_registry` deleted; `keys.py` added (about 70)                         |
| `conversion.py`       | 201         | about 190      | `_sorted_elements` instead of the public helper; the dataclass transformer is added                                               |
| `polars.py`           | 286         | about 260      | version gates instead of two `try/except TypeError`                                                                               |
| `__init__.py`         | 38          | about 55       | the lazy export                                                                                                                   |
| `recursive/key.py`    | 66          | 0              | moved (private) to `nested.keys`                                                                                                  |
| `utils/conversion.py` | 149         | about 115      | `_to_jsonable_recursive` and the dataclass branch deleted                                                                         |

The line counts are estimates from the prototype and from the code to delete; the gate is the
behaviour (sections 1 and 8), not the size. The docstrings are most of the lines of this package.
