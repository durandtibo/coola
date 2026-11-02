# Architecture of `coola.nested`

:book: This page describes the design of the `coola.nested` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/nested.md#properties); the API is in the [reference](../refs/nested.md).

## Overview

`coola.nested` has no traversal of its own: a function that goes inside the values uses the engine of
`recursive` (or the path iterator `dfs_items` of `iterator`) and describes one type, and a function
that works on the top level only is a plain loop. The key functions (`remove_keys_*`,
`add_prefix_suffix_to_keys`) use one private transformer for the mappings, registered in a copy of
the default registry; `to_flat_dict` joins the paths of `dfs_items`; `from_flat_dict` builds its
result with a private `dict` subclass and one final `recursive_apply`. The containers, the cycle check
and the depth are therefore the ones of the other packages.

## Constraints

1. **Simple and easy to maintain.** No function has a hand-written recursion or a registry of types
   built by hand: the traversal belongs to the engine of `recursive` or `iterator`, and `nested` only
   says what to do with a key or a container.
2. **Fast on the data that matters.** The benchmark gate holds on the data sizes that matter
   (thousands of nodes). A fixed cost of a few microseconds per call on a tiny input (the copy of the
   default registry) is accepted.
3. **Consistent with the siblings.** The same input is handled the same way by `equality`,
   `hashing`, `summary`, `iterator`, `recursive` and `nested`: the same containers are entered
   (dataclasses included), iterators and text are leaves, cycles are detected, no `RecursionError`,
   options are keyword-only.
4. **Optional libraries stay lazy.** `import coola.nested` imports no optional library. The polars
   function is loaded on first use, the way `coola.summary` does it (`lazy_exports`).

## Architecture in detail

```
coola.nested
  flat.py        to_flat_dict, from_flat_dict            dfs_items (iterator) + one transformer (_Node)
  keys.py        remove_keys_*, add_prefix_suffix_*      one private transformer (_KeyTransformer) + recursive_apply
  mapping.py     merge_mappings, flatten_mapping         plain loops over the top-level mappings, no nesting
  records.py     convert_to_dict_of_lists, ..._of_dicts  plain loops, no nesting
  polars.py      flatten_frame (+ private helpers)       loaded on first use
```

The rule: **a function that goes inside values uses the engine of `recursive` and describes one
type; a function that works on the top level only is a plain loop.** `merge_mappings`,
`flatten_mapping`, `convert_to_dict_of_lists` and `convert_to_list_of_dicts` do not look inside the
values. The conversion to JSON data is not part of `nested`: it is `coola.jsonable.to_jsonable`, a
registry of transformers on the engine of `recursive` with its own [design page](jsonable.md).

### `keys`: one private transformer

`_KeyTransformer` in `coola.nested.keys` is the only code that changes keys. `recursive` itself never
transforms keys (a property of its engine), so the one transformer that renames or drops keys lives in
the one package that needs it. A public `MapKeysTransformer` in `recursive` would become the right home
as soon as a second package needs it; moving a private class is a rename.

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
  `remove_keys_containing` and `remove_keys_starting_with` are thin wrappers (a predicate on `str`
  keys; a non-`str` key is kept).
- `add_prefix_suffix_to_keys(mapping, prefix="", suffix="", *, recursive=False)` is
  `_apply(mapping, rename=...)` when `recursive`, and one `rebuild_mapping` call on the top level
  otherwise (no engine needed). The common use (a prefix such as `train/` on the top-level metrics) is
  non-recursive, and the `remove_keys_*` functions are always recursive because removing a key at the
  top only is a dict comprehension. A rename that collides with another key raises `DuplicateKeyError`
  (the exception of the package), which is the rule for every function that can merge two entries and
  the only one that makes a rename function safe to expose.
- The registry is **derived from the default one**, so the containers are the ones of every other
  package: dataclasses are entered (the fields that are mappings are filtered or renamed, the
  dataclass is rebuilt), `MappingProxyType`, `deque` and the user-registered containers follow.
- The function of the values is the identity. A leaf costs one call of `_identity`, which the engine
  makes in about 0.1 µs; there is no measurable cost on trees.
- A copy of the default registry is 6 µs, which is below the noise of any call on a real structure.
  The derived registries are not cached: a cache would have to track the changes of the default
  registry (a user registers a transformer after the first call).
- The top-level type is kept (an `OrderedDict` gives an `OrderedDict`) and the annotation is a
  `TypeVar` bound to `Mapping`. A non-mapping input is a `TypeError` naming the type.

### `flat`

**`to_flat_dict(data, *, prefix=None, separator=".", to_str=None, escape=False)`** is built on
`dfs_items` of `coola.iterator`: the function is the join of the paths of the leaves.

- **What `nested` needs from `dfs_items`**: (1) the path is a tuple of raw labels from which the key,
  index or field name can be read (`str(label)` of a key is the flat segment; the escape is applied to
  it); (2) the keyword-only option `empty=True`, which also yields an empty container as `(path,
container)`, because an empty container is kept as the value of its key (`{"a": {}}` gives
  `{"a": {}}`) and the round trip must not lose it (without the option `dfs_items` follows the rule of
  the package: an empty container has no leaf); (3) the `registry` keyword of the other functions.
- **The containers are the ones of `iterator`**: mappings, sequences, deques, named tuples, ranges and
  dataclasses are entered; `str`, `bytes`, `bytearray` and iterators are leaves. Sets are the one
  documented exception (no index, so no label): the function uses a registry derived from the default
  one in which `set` and `frozenset` are leaves. Likewise each type of `to_str` is a leaf of that
  registry, and the leaf is written with `str()`.
- The cycle (`ValueError`), the depth (no `RecursionError`) and the order come from the engine of
  `iterator`.
- A key collision raises `ValueError`. `prefix`, `separator`, `to_str` and `escape` are keyword-only.
  The root being a leaf needs a `prefix` (`ValueError` otherwise). The root being empty gives `{}`.
  `prefix=""` is a prefix like any other (`".a"`), with no special case.

**`from_flat_dict(data, *, separator=".", restore_lists=False, escape=False)`.** The build loop
(`_set_nested`, `_split_key`) is flat, not recursive. Two devices avoid an id set and a recursion:

- The nodes created by the function are instances of a private `dict` subclass, `_Node`. A value given
  by the caller, even a `dict`, is a plain `dict`, so `isinstance(existing, _Node)` is the test "this
  dict is mine", and a dict of the caller is **never converted by `restore_lists`**.
- The result is built by a final `recursive_apply` with a registry derived from the default one in
  which `_Node` has a transformer: it rebuilds a plain `dict`, or a `list` when `restore_lists` is set
  and the keys are exactly `"0"`, ..., `"n-1"`. The root is a separate subclass whose transformer never
  builds a list (the root is always a `dict`). The depth is not limited (20,000 levels work).
- `data` is annotated `Mapping[str, Any]`.

### `mapping` and `records`

No new mechanism. These functions work on the top level only.

- **Options are keyword-only**: `merge_mappings(mappings, *, on_duplicate="raise", equal_nan=False)`,
  `flatten_mapping(mapping, *, on_duplicate="raise", always_prefix=False, separator=".",
equal_nan=False)`.
- **`merge_mappings(on_duplicate="suffix")` does not depend on the order.** A key of an input mapping
  that equals a generated suffixed key is a conflict whatever the position: the check is made once, on
  the set of all input keys, before the merge (`_free_suffixed_key` skips them). The result for
  `[{"a_1":3},{"a":1},{"a":2}]` is `{'a_1': 3, 'a': 1, 'a_2': 2}`, and the other order gives the same
  entries (the order of the keys follows the input). A generated key never overwrites or hides an
  input key, so there is nothing to raise.
- **`flatten_mapping(on_duplicate="prefix")` keeps the order of the input**: the first occurrence is
  renamed in place, so the input of the guide gives `{'train.loss': 1, 'acc': 2, 'val.loss': 3}`. Both
  functions can then say "the output follows the order of the input".
- **`convert_to_list_of_dicts`** rejects `Set` values with a `TypeError` naming the type, like it does
  for text, and accepts any `Sequence` (a generator is not one: `TypeError` that names the type).
  **`convert_to_dict_of_lists`** accepts any iterable of mappings (it materialises it once), so a
  generator works.
- **`get_first_value` lives in `coola.utils.mapping`**, next to `sort_by_keys` and `sort_by_values`: it
  does not look into a nested structure, so it is not a function of `nested`.

### `polars`: `flatten_frame`

- `flatten_frame(frame, *, separator=".", depth=None, expand_lists=False)` is exported lazily from
  `coola.nested` with `lazy_exports` (`from coola.nested import flatten_frame` works without importing
  polars at `import coola.nested`), and listed in `tests/integration/test_lazy_imports.py`.
- The helpers are private (`_is_nested_struct`, `_unnest_with_separator`, `_unnest_one_level`,
  `_expand_list_columns`, `_explode`); the tests of their behaviour go through `flatten_frame`.
- Only a `DataFrame` is accepted: a `LazyFrame` raises a `TypeError` that names the type and says
  "call `collect()`" (a lazy frame has no cheap schema, and the row count of an exploded list is only
  known after the evaluation). It is a visible rule instead of a warning at the end of a long chain.
- **The supported range is `polars>=1.0,<3.0`**, so two compatibility branches exist:
  `unnest(separator=...)` and `explode(empty_as_null=...)` are missing in the old versions. They are
  selected with `coola.utils.version.compare_version` on the version where each option appeared
  (`_UNNEST_SEPARATOR_VERSION`, `_EXPLODE_EMPTY_AS_NULL_VERSION`) instead of `except TypeError`, so a
  real `TypeError` propagates. Raising the minimum version would delete about 25 lines and two
  branches that CI cannot test on the current version.

## Public API

```python
from coola.nested import (
    DuplicateKeyError,
    add_prefix_suffix_to_keys,  # (mapping, prefix="", suffix="", *, recursive=False)
    convert_to_dict_of_lists,  # (records)
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

The names of the modules (`flat`, `keys`, `mapping`, `records`, `polars`) are not public contracts;
the package root is.

## Dependents

| Package                       | Use                                                                                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `coola.recursive`             | The engine, `get_default_registry`, `BaseTransformer` and `rebuild_mapping` (the public helper that `nested.keys` imports). |
| `coola.iterator`              | `dfs_items`, with the raw labels and the `empty` option.                                                                    |
| `coola.utils.mapping`         | `get_first_value`.                                                                                                          |
| `coola.io.json` (`save_json`) | The docstring points to `coola.jsonable.to_jsonable` for the conversion to JSON data.                                       |

## Design choices and trade-offs

**Dataclasses are entered by `to_flat_dict`, `remove_keys_*` and `add_prefix_suffix_to_keys`.**
`nested_data.md` says that a dataclass is a container in every package, and `recursive_apply` and
`to_jsonable` enter them: five different rules for the same object is the real cost. The price is that
a dataclass is not a leaf of the flat dict, and that a dataclass does not come back from
`from_flat_dict`. Keeping dataclasses as leaves of `nested` would need an exception per function in the
table of `nested_data.md`.

**The key transformer is private to `nested`.** `recursive` says "keys are never transformed"; a
transformer that renames or drops keys is the exception that only `nested` needs.

**`to_flat_dict` is built on `dfs_items`.** The labels are raw (a key, an index, a field name), and the
`empty` option keeps empty containers. A dedicated loop would be 25 lines, but would duplicate the
depth, the cycle and the container rules of the engine.

**A collision on rename raises.** The last entry winning would hide a loss of data.

## Not covered

- **Keys of `to_jsonable`.** A `stringify_keys` option (or a `str`/`int`/`float`/`bool`/`None` check
  that raises) would make the output strictly accepted by `json.dumps`. Natural next step with
  `_KeyTransformer`.
- **Strict JSON** (`NaN`, infinities): the contract is "accepted by the Python `json` module".
- **Dataclasses and pydantic models from `from_flat_dict`**: the function builds dicts and lists;
  rebuilding a typed object needs a schema and is not a flat-dict operation.
- **Lists of dicts to a flat table and back** (`pandas`/`polars` do it): `records` is two functions of
  the standard library.
- **An element-wise `rename_keys(data, func)` function**: `_KeyTransformer` supports it in one line;
  the public function can be added when a rename other than a prefix and a suffix is needed.
- **`merge_mappings` of nested mappings (deep merge).** Not a function of the package; the top-level
  merge with a duplicate strategy is what exists.
- **Parallel or lazy application**: the outputs are built whole, as in `recursive`.
