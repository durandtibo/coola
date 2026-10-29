# Architecture of `coola.nested`

:book: This page describes the design of the `coola.nested` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../../uguide/nested.md#properties); the API is in the [reference](../../refs/nested.md).

## Overview

`coola.nested` has no traversal of its own: a function that goes inside the values uses the engine of
`recursive` (or the path iterator `dfs_items` of `iterator`) and describes one type, and a function
that works on the top level only is a plain loop. The key functions (`remove_keys_*`,
`add_prefix_suffix_to_keys`) use one private transformer for the mappings, registered in a copy of
the default registry; `convert_to_jsonable` adds a transformer for the sets (a sorted list) and one
for the dataclasses (a `dict`); `to_flat_dict` joins the paths of `dfs_items`; `from_flat_dict`
builds its result with a private `dict` subclass and one final `recursive_apply`. The containers,
the cycle check and the depth are therefore the ones of the other packages.

## Constraints

1. **Simple and easy to maintain.** The package is 1,543 lines (and 2,974 lines of tests) for 14
   functions. Four of them contain a hand-written recursion (`_flatten_into`, `_restore_lists`,
   `_rename_nested_keys`, and `to_jsonable._to_jsonable_recursive` that `convert_to_jsonable`
   depends on), a fifth builds a registry of ten types by hand. The target deletes these instead of
   adding code next to them: the traversal belongs to the engine of `recursive`, and `nested` only
   says what to do with a key or a container.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The release notes and the migration page list the changes.
3. **No slower than before.** The benchmark gate must hold on the data sizes that matter
   (thousands of nodes). A fixed cost of a few microseconds per call on a tiny input is accepted and
   stated.
4. **Consistent with the siblings.** The same input must be handled the same way by `equality`,
   `hashing`, `summary`, `iterator`, `recursive` and `nested`: the same containers are entered
   (dataclasses included), iterators and text are leaves, cycles are detected, no `RecursionError`,
   options are keyword-only.
5. **Optional libraries stay lazy.** `import coola.nested` imports no optional library. The polars
   function is loaded on first use, the way `coola.summary` does it (`lazy_exports`).

## Architecture in detail

```
coola.nested
  flat.py        to_flat_dict, from_flat_dict            dfs_items (iterator) + one transformer (_Node)
  keys.py        remove_keys_*, add_prefix_suffix_*      one private transformer (_KeyTransformer) + recursive_apply
  mappings.py    merge_mappings, flatten_mapping         plain loops over the top-level mappings, no nesting
  records.py     convert_to_dict_of_lists, ..._of_dicts  plain loops, no nesting
  json.py        convert_to_jsonable                     recursive_apply with two transformers (dataclass, set)
  polars.py      flatten_frame (+ private helpers)       loaded on first use
```

The rule: **a function that goes inside values uses the engine of `recursive` and describes one
type; a function that works on the top level only is a plain loop.** `merge_mappings`,
`flatten_mapping`, `convert_to_dict_of_lists` and `convert_to_list_of_dicts` do not look inside the
values; they are right as they are, apart from the points of. The modules are renamed to
the plural of their content (`mapping.py` becomes `mappings.py`, `conversion.py` is split) only if the
split of is done; the names of the functions do not depend on it.

### `keys`: one private transformer

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
  (impossible before for a prefix and a suffix because the map is injective on `str`, but the
  transformer is generic and a rename function is one step away).
- The registry is **derived from the default one**, so the containers are the ones of every other
  package: dataclasses are entered (the fields that are mappings are filtered or renamed, the
  dataclass is rebuilt), `MappingProxyType`, `deque` and the user-registered containers follow. The
  list of ten types is deleted.
- The function of the values is the identity. A leaf costs one call of `_identity`, which the engine
  makes in about 0.1 µs; the prototype below shows no measurable cost on trees.
- The top-level type is kept (an `OrderedDict` gives an `OrderedDict`) and the annotation is a
  `TypeVar` bound to `Mapping`, so `-> dict` stops lying. A non-mapping input is a `TypeError`
  naming the type (before: `remove_keys_if([1], ...)` returns the list).
- `KeyFilterTransformer` is deleted from `coola.recursive` (`__all__`, guide, tests): `recursive`
  recovers the property "keys are never transformed" without an exception, and the `recursive` design
  ("`MapKeysTransformer` is the natural next transformer") is answered here, in the one
  package that needs it.

### `flat`

**`to_flat_dict(data, *, prefix=None, separator=".", to_str=None, escape=False)`** is built on
`dfs_items` of `coola.iterator`.
`_flatten_into`, `_check_no_cycle`, `_build_key` and the recursion are deleted: the function is the
join of the paths of the leaves.

- **What `nested` needs from `dfs_items`** (to put in the design of the `iterator` design D9, which only
  says "the path of each leaf"): (1) the path is a tuple of labels from which the raw key, index or
  field name can be read (`str(label)` of a key is the flat segment; the escape is applied to it);
  (2) a keyword-only option `empty=True` that also yields an empty container, as `(path, container)`,
  because an empty container is kept as the value of its key (`{"a": {}}` gives `{"a": {}}`) and the
  round trip must not lose it (without the option `dfs_items` follows the rule of the package: an
  empty container has no leaf); (3) the `registry` keyword of the other functions.
- **The containers are the ones of `iterator`**: mappings, sequences, deques, named tuples, ranges and
  dataclasses are entered (behaviour change for the dataclasses, ); `str`, `bytes`, `bytearray` and
  iterators are leaves. Sets are the one documented exception (no index, so no label): the function
  uses a registry derived from the default one in which `set` and `frozenset` are leaves. Likewise
  each type of `to_str` is a leaf of that registry, and the leaf is written with `str()`.
- The cycle (`ValueError`), the depth (no `RecursionError`) and the order come from the engine of
  `iterator`, which `to_flat_dict` no longer duplicates.
- A key collision raises `ValueError` (kept). `to_str` is kept; `prefix`, `separator`, `to_str` and
  `escape` are keyword-only. The root being a leaf needs a `prefix`, as before (`ValueError`
  otherwise). The root being empty gives `{}`.
- `prefix=""` is a prefix like any other (`".a"`). Documented; no special case.

**`from_flat_dict(data, *, separator=".", restore_lists=False, escape=False)`.** The build loop
(`_set_nested`, `_split_key`) stays: it is flat, not recursive. Two changes remove the id set and the
recursion:

- The nodes created by the function are instances of a private `dict` subclass, `_Node`. A value
  given by the caller, even a `dict`, is a plain `dict`, so `isinstance(existing, _Node)` is the test
  "this dict is mine" (it replaces the set `created` of ids) and a dict of the caller is **never
  converted by `restore_lists`**.
- The result is built by a final `recursive_apply` with a registry derived from the default one in
  which `_Node` has a transformer: it rebuilds a plain `dict`, or a `list` when `restore_lists` is set
  and the keys are exactly `"0"`, ..., `"n-1"`. The root is a separate subclass whose transformer
  never builds a list (the root is always a `dict`). `_restore_lists` is deleted, and the depth is no
  longer limited. The prototype gives the same output as the previous version on the benchmark
  tree and runs on 20,000 levels.
- `data` is annotated `Mapping[str, Any]`.

### `json`: `convert_to_jsonable` and `to_jsonable`

`convert_to_jsonable(data)` is `recursive_apply(data, to_jsonable, registry=registry)` with a registry
derived from the default one in which:

- `set` and `frozenset` have a transformer whose `rebuild` returns a sorted list (kept from
  `SetToListTransformer`, now private), sorted with a key that does not depend on the iteration order:
  the elements when they compare, otherwise `(type name, repr)`. `try_sorted_list` becomes the private
  `_sorted_elements` of this module;
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

`convert_to_jsonable` documents what it does not do (keys, `NaN`, tuples) as before. A
`stringify_keys` option is the obvious next step and is not part of this design.

### `mappings` and `records`: small corrections

No new mechanism. These functions work on the top level only.

- **Options keyword-only**: `merge_mappings(mappings, *, on_duplicate="raise", equal_nan=False)`,
  `flatten_mapping(mapping, *, on_duplicate="raise", always_prefix=False, separator=".",
equal_nan=False)`.
- **`merge_mappings(on_duplicate="suffix")` does not depend on the order**. A key of an input
  mapping that equals a generated suffixed key is a conflict whatever the position: the check is made
  once, on the set of all input keys, before the merge (a `set` of the keys of every mapping, then
  `_free_suffixed_key` skips them). The result for `[{"a_1":3},{"a":1},{"a":2}]` is
  `{'a_1': 3, 'a': 1, 'a_2': 2}` and the result for the other order is the same entries (the order of
  the keys follows the input). This removes the `raise` of "it collides with a suffixed key": a
  generated key never overwrites or hides an input key, so there is nothing to raise. The error of the
  docstring (`DuplicateKeyError` "if a key of an input mapping is equal to a key generated by the
  suffix strategy") goes away. The unit tests of the old error become tests of the new result.
- **`flatten_mapping(on_duplicate="prefix")` keeps the order of the input**: the first
  occurrence is renamed in place (the dict is rebuilt at the end instead of `pop` then re-adding), so
  `{'train.loss': 1, 'acc': 2, 'val.loss': 3}` for the input of the guide. Both functions can then
  say "the output follows the order of the input". Behaviour change, documented.
- **`convert_to_list_of_dicts`** rejects `Set` values with a `TypeError` naming the type, like it
  already does for text, and accepts any `Sequence` (a generator is not one: `TypeError` that
  names the type instead of the `len` error). **`convert_to_dict_of_lists`** accepts any iterable of
  mappings (it materialises it once), so a generator works.
- **`get_first_value` moves to `coola.utils.mapping`**, next to `sort_by_keys` and
  `sort_by_values`: it does not look into a nested structure, so it is not a function of `nested`.
  The signature, the `ValueError` on an empty mapping, the docstring and the tests move with it;
  `coola.nested` stops exporting it (breaking).

### `polars`: `flatten_frame`

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
- **The supported range stays `polars>=1.0,<3.0`**, so the two compatibility branches
  stay (`unnest(separator=...)` and `explode(empty_as_null=...)` are missing in the old versions).
  They are selected with `coola.utils.version.compare_version` on the version where each option
  appeared (looked up in the polars changelog in and tested with the oldest version in the
  CI matrix) instead of `except TypeError`, so a real `TypeError` propagates.

## Public API

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

Moved: `get_first_value` to `coola.utils.mapping`. Removed: `try_sorted_list` (private), `SetToListTransformer` (private),
`coola.nested.polars.{is_nested_struct, unnest_with_separator, unnest_one_level, expand_list_columns}`
(private), `coola.recursive.KeyFilterTransformer` (moved, private), the positional options, the
dataclass branch of `coola.utils.conversion.to_jsonable`. The names of the modules (`flat`, `mapping`,
`conversion`, `polars`) are not public contracts (the package root is) and can be kept to keep the
diff of each step small; the split of is optional.

## Dependents

| Package                                         | Change                                                                                                                                                                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.recursive`                               | `KeyFilterTransformer` and `key.py` deleted (guide, `__all__`, tests). `rebuild_mapping` and `rebuild_sequence` stay the public helpers (`nested.keys` imports `rebuild_mapping`). The **Properties** of `recursive` lose the implicit exception. |
| `coola.utils.conversion.to_jsonable`            | The dataclass branch and `_to_jsonable_recursive` are deleted; the docstring points to `convert_to_jsonable`.                                                                                                                                     |
| `coola.io.json` (`save_json`)                   | The docstring already points to `convert_to_jsonable`; no code change.                                                                                                                                                                            |
| `tests/integration/recursive/test_dataclass.py` | `test_remove_keys_if_dataclass_is_a_leaf` is replaced by `test_remove_keys_if_enters_the_dataclass`; `test_convert_to_jsonable_dataclass_is_converted_whole` stays and gains the set and the tuple of .                                           |
| `tests/package_checks.py`                       | Unchanged (`coola.nested.to_flat_dict`).                                                                                                                                                                                                          |
| `iterator`                                      | **Prerequisite:** `dfs_items` with the labels and the `empty` option of (D9 of the `iterator` design).                                                                                                                                            |

## Decisions for the maintainers

**D1. Delete `get_first_value`?** _Recommended: yes._ It is a one-liner without user in `src`, and the
package shrinks to functions that do real work. The cost is one line for the callers
(`next(iter(data.values()))`, which raises `StopIteration` on an empty mapping). Alternative: keep it;
it costs nothing to maintain, but it is the only function of the package that nobody needs from a
nested structure.

Decision: move it to `coola.utils.mapping` (not deleted). Reflected in, the API list and.

**D2. Own the key transformer in `nested`, or generalise it in `recursive`?** _Recommended: in
`nested`, private._ `recursive` says "keys are never transformed", and a transformer that renames or
drops keys is the exception that only `nested` needs. Alternative: a public `MapKeysTransformer` in
`recursive`, which the `recursive` design foresees. It becomes the right home as soon as a
second package needs it; moving a private class is a rename.

Decision: follow recommendation

**D3. Raise the minimum polars version?** _Recommended: look at the support policy of
`pyproject.toml` (`polars>=1.0,<3.0`) when is made._ If the oldest supported version has
`unnest(separator=...)` and `explode(empty_as_null=...)`, the two compatibility branches
(`unnest_with_separator`, `_explode`) are deleted, about 25 lines and two untestable branches on
current CI. If not, keep them behind `compare_version`.

Decision: keep `polars>=1.0,<3.0`. The compatibility branches stay, behind `compare_version`.

**D4. Enter dataclasses in `to_flat_dict`, `remove_keys_*` and `add_prefix_suffix_to_keys`?**
_Recommended: yes._ `nested_data.md` says that a dataclass is a container in every package and
`recursive_apply` and `convert_to_jsonable` enter them already; five different rules for the same
object is the real cost. The price is a behaviour change for code that relied on a dataclass
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
(25 lines), `dfs_items` later._ The path-aware iterator is not built, the loop is small
and fixes the depth, the cycle and the dataclass before, and deleting it when `dfs_items` lands is a
local change. Alternative: make `dfs_items` first and keep `to_flat_dict` as is; it blocks the fix on
an unrelated feature.

Decision: make `dfs_items` first. `to_flat_dict` is built on it; the work is split in (iterator) and (nested). The design of `dfs_items` has two requirements from `nested`: readable raw labels and an `empty` option.

## Not covered

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
