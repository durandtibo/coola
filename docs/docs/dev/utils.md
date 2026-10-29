# Architecture of `coola.utils`

:book: This page describes the design of the `coola.utils` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/utils.md#properties); the API is in the [reference](../refs/utils.md).

## Constraints

1. **Simple and easy to maintain.** `coola.utils` is 6,048 lines in 29 modules and two subpackages
   (`imports` 1,742, `fallback` 421, the 29 modules 3,885), tested by 9,196 lines. It mixes the
   foundation of `coola` (imports, singletons, formatting) with standalone helpers (Bloom filter,
   LRU cache, git, dates, strings) that the project accumulated as it evolved. The target separates
   the two, removes the repetition (the same four functions written eleven times) and deletes only
   what is risky or obsolete. The useful helpers are **kept**, with their tests.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias, no
   deprecation period (a moved module is imported from its new path). The release
   notes and the migration page (`docs/docs/migration/v1.1_to_v2.0.md`) list the changes.
3. **The rules of the repository apply to the helpers too.** Options are keyword-only, an argument
   check goes through `coola.validation`, a shared helper is public and tested, a lazily built
   process-wide object is a `LazySingleton`. Today 10 of the surviving functions break one of the
   first two rules.
4. **A leaf of the layering.** `utils` is below every other package. It may import the standard
   library and `coola.validation`, nothing else. `registry` depends on `utils`, never the
   opposite. The new `formatting` package sits above it (it may import `utils` and `validation`), and the new
   `tools` package sits next to `display` in the core group: it may import `utils`, `validation`,
   `formatting` and `display`, and no feature package.
5. **Optional libraries stay lazy.** Importing `coola.utils` or `coola.utils.imports` imports no
   optional library, and the availability helpers are the one mechanism the other packages use to
   ask whether a library is there.

## Architecture

### Layout

```
coola/utils/                      the foundation, about 2,030 lines (6,048 before for all of it)
├── __init__.py                   docstring only, no re-export
├── fork.py                       track_fork, ForkAware
├── integer.py                    int_repr
├── introspection.py              get_qualified_class_name, get_type_constructor, is_same_type,
│                                 supports_methods, has_generated_dataclass_eq,
│                                 get_all_child_classes, is_lambda_function, check_not_lambda_function
├── mapping.py                    sort_by_keys, sort_by_values, get_first_value
├── path.py                       add_uuid_suffix, sanitize_path, find_root_package_parent
├── safe.py                       safe_repr, safe_str
├── selection.py                  smallest_items
├── singleton.py                  LazySingleton
├── stats.py                      quantile
├── version.py                    compare_version, get_package_version
├── imports/
│   ├── __init__.py               re-exports the public names
│   ├── package.py                MissingPackageError, availability_cache, clear_availability_caches,
│   │                             package_available, module_available, check_package,
│   │                             raise_package_missing_error          (was universal.py)
│   ├── optional.py               is_<pkg>_available and check_<pkg> for the 11 libraries, and
│   │                             is_torch_numpy_available / check_torch_numpy
│   └── lazy.py                   lazy_exports, is_module_imported
└── fallback/                     one fake module per library; factory.py takes the package name

coola/formatting/                 turning values into text, about 810 lines, moved as they are
├── __init__.py                   docstring only
├── structure.py                  str_indent, repr_indent, str_mapping, repr_mapping,
│                                 str_mapping_line, repr_mapping_line, str_sequence, repr_sequence,
│                                 str_sequence_line, repr_sequence_line   (from utils.format)
├── quantity.py                   str_human_byte_size, find_best_byte_unit, str_time_human, make_bar
│                                                                          (from utils.format)
├── text.py                       char_diff_summary, count_lines, remove_empty_lines, slugify,
│                                 truncate_str                            (was utils.string)
├── secret.py                     mask_secret                             (was utils.secret)
└── diff.py                       unified_diff                            (was utils.text_diff)

coola/tools/                      general-purpose helpers, about 1,510 lines, moved as they are
├── __init__.py                   docstring only
├── array.py                      to_array                       (was utils.array)
├── tensor.py                     to_tensor, is_cuda_available, is_mps_available,
│                                 get_available_devices          (was utils.tensor)
├── batching.py                   batchify
├── bloom_filter.py               BloomFilter
├── date.py                       get_today_date, to_date, to_datetime
├── deprecation.py                deprecate_positional_options
├── env_vars.py                   check_env_vars, get_required_env_var, temp_env_vars
├── file_size.py                  get_file_size
├── git.py                        run_git_command, get_current_branch, get_current_commit_hash,
│                                 is_git_repo
├── logging.py                    CaptureHandler
├── lru.py                        LRUCache
├── password.py                   get_password
├── recursion.py                  raise_recursion_limit_error
└── timing.py                     TimingResult, timeblock
```

`formatting` is a core package next to `utils`: every function that turns a value into a string
for a message, a `repr` or a report is there (before: they are spread over `utils.format`,
`utils.string`, `utils.secret` and `utils.text_diff`, and `display` and `registry` import the first).
It may import `utils` (`sort_by_keys`) and `validation`, and no other package. `display`, `registry`,
`tools.timing` and the feature packages import from it. The name is decision D1b: `coola.format`
would shadow the builtin in `from coola import format`.

The move into `formatting` and `tools` is a rename (plus the split of `format.py` in two modules by
theme): the modules keep their code, their tests (`tests/unit/formatting/test_<module>.py`,
`tests/unit/tools/test_<module>.py`) and their docstrings; only the import path changes. The modules
of `tools` are the ones that `coola` itself does not import. They stay public and documented
(`docs/docs/uguide/tools.md`, `refs/tools.md`). The name `tools` is decision D1.

Moved elsewhere: `coola/registry/default.py` (`make_default_registry_singleton`,
`reset_default_registries`, `temporary_default_registries`), `coola/registry/plugins.py`
(`load_registry_plugins`, `get_loaded_plugins`, `LoadedPlugin`, the two environment variable names)
and `coola/nested/conversion.py` (`to_jsonable`, which only `nested` calls and whose docstring sends
the reader to `nested`). Deleted: two items with a reason each.

### Disposition of every module

"Callers" are the packages of `src/coola` that import the name (outside `utils`), from the census of
. "Test lines" are the lines of `tests/unit/utils/test_<module>.py`. **Move** is a rename
to `coola.tools` (or `coola.formatting` where stated) with its tests.

| Module          | Lines (test lines) | Callers before                                                                 | Target                                                                                                                                                                                     |
| --------------- | ------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `array`         | 58 (81)            | none                                                                           | **Move** to `tools`.                                                                                                                                                                       |
| `batching`      | 51 (101)           | none                                                                           | **Move.**                                                                                                                                                                                  |
| `bloom_filter`  | 213 (174)          | none                                                                           | **Move.**                                                                                                                                                                                  |
| `conversion`    | 115 (211)          | `nested`                                                                       | **Move** to `nested/conversion.py`.                                                                                                                                                        |
| `date`          | 99 (139)           | none                                                                           | **Move.**                                                                                                                                                                                  |
| `deprecation`   | 112 (97)           | none                                                                           | **Move.**                                                                                                                                                                                  |
| `env_vars`      | 188 (518)          | none                                                                           | **Move.** `temp_env_vars` changes `os.environ` and says so (not thread-safe); it stays because it is tested, documented and the standard `monkeypatch` is not available outside pytest.    |
| `file_size`     | 55 (74)            | none                                                                           | **Move.**                                                                                                                                                                                  |
| `fork`          | 76 (50)            | `identifier`, `registry`, `singleton`                                          | **Keep.**                                                                                                                                                                                  |
| `format`        | 491 (587)          | `display`, `registry`                                                          | **Move** to `formatting` (`structure.py`, `quantity.py`); keep all 14 functions; options keyword-only; `validate_ge` for `num_spaces`. `tools.timing` imports `str_time_human` from there. |
| `git`           | 153 (271)          | none                                                                           | **Move.** The subprocess call is already hardened (resolved executable, no shell, documented warning).                                                                                     |
| `integer`       | 48 (64)            | `hashing`                                                                      | **Keep, fix the sign** of a huge negative integer.                                                                                                                                         |
| `introspection` | 322 (514)          | `equality`, `hashing`, `io`, `random`, `recursive`                             | **Keep**, but delete `get_fully_qualified_name` (: two functions for one concept).                                                                                                         |
| `logging`       | 50 (167)           | none                                                                           | **Move.**                                                                                                                                                                                  |
| `lru`           | 204 (308)          | none                                                                           | **Move.**                                                                                                                                                                                  |
| `mapping`       | 94 (145)           | `display`, `format`                                                            | **Keep.**                                                                                                                                                                                  |
| `password`      | 69 (233)           | none                                                                           | **Move** to `tools` (D2: kept for now). It reads and validates credentials; `coola` has no use for it, so it is the first candidate to remove if it is reported as a risk.                 |
| `path`          | 212 (396)          | `io`                                                                           | **Keep**, delete `working_directory` (risk (b)): `contextlib.chdir` is the same and the docstring already warns it is not thread-safe.                                                     |
| `recursion`     | 46 (37)            | none                                                                           | **Move** to `tools` (D2: kept for now). `equality`, `recursive` and `iterator` use explicit stacks, so nothing in `coola` raises the `RecursionError` it explains.                         |
| `safe`          | 76 (126)           | `equality`, `summary`                                                          | **Keep.**                                                                                                                                                                                  |
| `secret`        | 48 (55)            | none                                                                           | **Move** to `formatting`.                                                                                                                                                                  |
| `selection`     | 46 (58)            | `summary`                                                                      | **Keep, fix the iterator and make `limit` keyword-only with `validate_ge`**.                                                                                                               |
| `singleton`     | 428 (729)          | `equality`, `hashing`, `iterator`, `random`, `recursive`, `summary`, `testing` | **Split**: `LazySingleton` stays; the default-registry and plugin code moves to `registry`.                                                                                                |
| `stats`         | 64 (146)           | `reducer`                                                                      | **Keep**; the quantile check uses `validate_range`.                                                                                                                                        |
| `string`        | 203 (224)          | none                                                                           | **Move** to `formatting` (`text.py`).                                                                                                                                                      |
| `tensor`        | 123 (183)          | `reducer` (`to_tensor`), `testing` (`is_cuda_available`, `is_mps_available`)   | **Move** to `tools` (they import NumPy and PyTorch when installed, ); `reducer` and `testing` import from `coola.tools.tensor`.                                                            |
| `text_diff`     | 69 (129)           | none                                                                           | **Move** to `formatting` (`diff.py`).                                                                                                                                                      |
| `timing`        | 93 (181)           | none                                                                           | **Move** (its import of `display` is legal in `tools`, and it ends the cycle of ).                                                                                                         |
| `version`       | 73 (97)            | `nested`                                                                       | **Keep.** `get_package_version` is the building block of `compare_version`.                                                                                                                |
| `imports`       | 1,742 (1,897)      | 59 files                                                                       | **Rewrite** as `package.py`, `optional.py`, `lazy.py`.                                                                                                                                     |
| `fallback`      | 421 (579)          | 39 files                                                                       | **Keep the shape**: one module per library; `factory.py` takes the package name instead of a `raise_error` callable.                                                                       |

### `imports`: one definition of "available"

The current design copies one 100-line module per library. The four functions are the same except
for a string, and two of them are unused or harmful. The target keeps what the 59 files that import it use and
nothing else.

`package.py` (was `universal.py`) defines the mechanism once, with one message:

```python
class MissingPackageError(ModuleNotFoundError, RuntimeError): ...  # unchanged


def availability_cache(fn: F) -> F: ...  # unchanged
def clear_availability_caches() -> None: ...  # unchanged
def package_available(name: str) -> bool: ...  # installed, not imported (find_spec)
def module_available(name: str) -> bool: ...  # importable, imported, cached


def check_package(package: str, *, install: str | None = None) -> None:
    """Raise ``MissingPackageError`` unless ``module_available(package)``."""


def raise_package_missing_error(package: str, *, install: str | None = None) -> Never:
    """The one message: ``'x' package is required ...`` with ``pip install`` and ``uv pip install``.
    ``install`` is the pip name when it differs from the import name (the default)."""
```

`optional.py` is the only place that names a library. For each library, two functions with the same
two lines:

```python
@availability_cache
def is_numpy_available() -> bool:
    r"""Indicate if ``numpy`` is installed and can be imported. ..."""
    return module_available("numpy")


def check_numpy() -> None:
    r"""Check that ``numpy`` is available. ..."""
    check_package("numpy")
```

so that `check_numpy()` is by definition `check_package("numpy")`, and `is_numpy_available` keeps
its name for the 59 files that import it and for the `if TYPE_CHECKING or is_numpy_available():`
idiom, which needs a statically known name. The torch and NumPy pair keeps its compatibility check
(`torch.tensor([1.0]).numpy()`), imported inside the function.

Deleted from the public surface: the eleven decorators `x_available` and `torch_numpy_available`
(no caller, silent `None`), `decorator_package_available`, the eleven `raise_x_missing_error`
(the fallback modules call `raise_package_missing_error` through `make_fake_class("numpy")`),
`LazyModule`, `lazy_import` and `import_object` (replaced by `pkgutil.resolve_name`, ). The
`check_package` signature loses its positional `command`: the `install` keyword is the pip name, and
the message is the same as for every other package.

`lazy.py` keeps `lazy_exports` (eight packages) and `is_module_imported` (`factory`).

Adding a library (`docs/docs/dev/add_backend.md`) is two functions in `optional.py`, their names in
`imports/__init__.py`, one fake module in `fallback/<pkg>.py` and one row in `testing/marks.py`, plus
`pyproject.toml`: about 25 added lines instead of a 100-line module copied from `pandas.py`.

### `LazySingleton` and the default registries

`utils/singleton.py` keeps the primitive and makes the two private test hooks public, so that the
code that moves out does not need the privates:

```python
class LazySingleton(Generic[T]):
    def get(self) -> T: ...  # unchanged
    def is_built(self) -> bool: ...  # unchanged
    def reset(self) -> None:  # raises RuntimeError from the factory's own thread
        ...

    def set(self, value: T) -> None:  # new: replaces the cached value, under the lock
        ...
```

`coola/registry/default.py` receives `make_default_registry_singleton(registry_cls,
register_defaults, *, plugin_group=None)`, `reset_default_registries()` and
`temporary_default_registries()`; the last one is written with `is_built`, `get`, `set` and
`reset`: on entry, `set(get().copy())` for each registry that is built; on exit, `set(original)` for
those registries and `reset()` for the ones that were built inside the context. The module
already knows what a registry is (it calls `copy`), so its place is `coola.registry`.
`coola/registry/plugins.py` receives the entry-point loader, `LoadedPlugin`, `get_loaded_plugins` and
the two environment variable names (`COOLA_STRICT_PLUGINS`, `COOLA_DISABLE_PLUGINS`) with their
two-line `_env_flag`. Behaviour is unchanged. The six feature packages and `coola.testing` import
from `coola.registry` instead of `coola.utils.singleton`; `coola.random` already depends on
`coola.registry` (its registry is a `Registry`).

### Contracts of the helpers that stay

| Function                                       | Signature after the change                                | Change                                                                                         |
| ---------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `str_indent`, `repr_indent`                    | `(original, *, num_spaces=2)`                             | keyword-only; `validate_ge(num_spaces, 0, name="num_spaces")`                                  |
| `str_mapping`, `repr_mapping`                  | `(mapping, *, sorted_keys=False, num_spaces=2)`           | keyword-only                                                                                   |
| `str_mapping_line`, `repr_mapping_line`        | `(mapping, *, sorted_keys=False, separator=", ")`         | keyword-only                                                                                   |
| `smallest_items`                               | `(data: Collection, *, limit=None)`                       | a collection (it is iterated twice when the items cannot be compared); `validate_ge(limit, 0)` |
| `int_repr`                                     | `(value: int)`                                            | `-0x…` for a huge negative integer (a hash change, decision D5)                                |
| `quantile`                                     | `(values, quantiles)`                                     | each quantile checked by `validate_range(q, 0.0, 1.0, name="quantiles")`                       |
| `check_package`, `raise_package_missing_error` | `(package, *, install=None)`                              | one message, keyword-only `install`                                                            |
| `make_default_registry_singleton`              | `(registry_cls, register_defaults, *, plugin_group=None)` | keyword-only; moves to `coola.registry`                                                        |
| `LazySingleton.reset`                          | `() -> None`                                              | raises `RuntimeError` when called by the thread that builds the value                          |
| `compare_version`                              | `(package, op, version)`                                  | none                                                                                           |

### Rule for a new helper

A helper used by two or more packages of `coola` goes to `coola.utils` (`CLAUDE.md`), public and
tested. A general-purpose helper that `coola` does not use goes to `coola.tools`. Both are generic
, and a helper that is a step of one feature lives in that feature. The `ast` test of
keeps `utils` honest: when the last caller of a `utils` module goes away, the test fails
and the module moves to `tools`.

## Public API

`coola.utils` (the modules keep all their functions except the ones named in):

```python
from coola.utils.fork import ForkAware, track_fork
from coola.utils.imports import (  # same names as before, minus the removed ones
    MissingPackageError,
    availability_cache,
    check_package,
    clear_availability_caches,
    is_module_imported,
    lazy_exports,
    module_available,
    package_available,
    raise_package_missing_error,
    # for each of colorlog, jax, numpy, packaging, pandas, polars, pyarrow, pydantic, rich,
    # torch, xarray: is_<pkg>_available, check_<pkg>; and is_torch_numpy_available,
    # check_torch_numpy
)
from coola.utils.integer import int_repr
from coola.utils.introspection import (
    check_not_lambda_function,
    get_all_child_classes,
    get_qualified_class_name,
    get_type_constructor,
    has_generated_dataclass_eq,
    is_lambda_function,
    is_same_type,
    supports_methods,
)
from coola.utils.mapping import get_first_value, sort_by_keys, sort_by_values
from coola.utils.path import add_uuid_suffix, find_root_package_parent, sanitize_path
from coola.utils.safe import safe_repr, safe_str
from coola.utils.selection import smallest_items
from coola.utils.singleton import LazySingleton
from coola.utils.stats import quantile
from coola.utils.version import compare_version, get_package_version
```

`coola.formatting` (the 14 functions of `format`, `str_*` and `repr_*` in `structure`, the byte, time and bar functions in `quantity`, with options keyword-only; `text.{char_diff_summary, count_lines, remove_empty_lines, slugify, truncate_str}`, `secret.mask_secret`, `diff.unified_diff`).

`coola.tools` (import paths change, the functions do not): `coola.tools.array.to_array`,
`coola.tools.tensor.{to_tensor, is_cuda_available, is_mps_available, get_available_devices}`,
`coola.tools.batching.batchify`, `coola.tools.bloom_filter.BloomFilter`,
`coola.tools.date.{get_today_date, to_date, to_datetime}`,
`coola.tools.deprecation.deprecate_positional_options`,
`coola.tools.env_vars.{check_env_vars, get_required_env_var, temp_env_vars}`,
`coola.tools.file_size.get_file_size`, `coola.tools.git.*`, `coola.tools.logging.CaptureHandler`,
`coola.tools.lru.LRUCache`, `coola.tools.password.get_password`,
`coola.tools.recursion.raise_recursion_limit_error`, `coola.tools.timing.{TimingResult, timeblock}`.

Of the 82 public names of the 29 modules, 24 stay in `utils`, 26 move to `tools`, 21 move to `formatting`, 9 move to
`registry` (8) and `nested` (1), and 2 are deleted (`get_fully_qualified_name`, `working_directory`). `coola.utils.imports` goes from 60 public names to 33: `is_x_available` and `check_x` for the eleven
libraries and the torch/NumPy pair (24), the seven names of `package.py` and the two of `lazy.py`.
The 27 that go are the 12 decorators, the 11 `raise_x_missing_error`, `decorator_package_available`,
`LazyModule` and `lazy_import`, and `import_object`.

## Dependents

| Package                                                             | Change                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `display`, `registry`                                               | Import the formatting functions from `coola.formatting.structure`. None for the keyword-only options: no call site passes `sorted_keys`, `num_spaces` or `separator`, and `mapping_line_fn=str_mapping_line` passes the function. The import paths are unchanged.            |
| `equality`, `hashing`, `iterator`, `random`, `recursive`, `summary` | `from coola.registry.default import make_default_registry_singleton` (and `LazySingleton` from `coola.utils.singleton`); `plugin_group=` is already a keyword at the six call sites.                                                                                         |
| `testing`                                                           | `temporary_default_registries` from `coola.registry.default`; `is_cuda_available` and `is_mps_available` from `coola.tools.tensor` (the lazy `import_module` of the module name stays, so a test module does not import PyTorch).                                            |
| `reducer`                                                           | `to_tensor` from `coola.tools.tensor`; `reducer/utils.py` and `rounding/interface.py` call `pkgutil.resolve_name("pkg.mod:name")`; `registry/type.py` too.                                                                                                                   |
| `recursive`                                                         | The rebuild error uses `get_qualified_class_name(type(data))`.                                                                                                                                                                                                               |
| `nested`                                                            | `nested/conversion.py` owns the leaf converter that was `utils.conversion.to_jsonable`.                                                                                                                                                                                      |
| `hashing`                                                           | `int_repr` returns `-0x…` for a negative integer with more than 4,300 digits: the hash of such an integer changes.                                                                                                                                                           |
| `io`, `identifier`, `factory`, `equality`, `random`, `summary`      | No change: `add_uuid_suffix`, `sanitize_path`, `track_fork`, `is_module_imported`, `supports_methods`, `has_generated_dataclass_eq`, `is_same_type`, `safe_str`, `smallest_items` keep their names. The two calls of `smallest_items` in `summary` pass `limit=` by keyword. |
| 15 test files outside `tests/unit/utils`                            | 21 `patch("coola.utils.imports.<pkg>.is_<pkg>_available", ...)` become `patch("coola.utils.imports.optional.is_<pkg>_available", ...)`. The tests of the moved modules move to `tests/unit/tools` with their code.                                                           |

## Decisions for the maintainers

Status: all decisions are taken. The recommendation was followed everywhere except two: D2 (only
`working_directory` is removed; `password` and `recursion` are kept, in `tools`) and D4 (no census
test for now).

**D1. Name and shape of the new sub-package.** _Recommended: `coola.tools`, one module per theme,
the modules moved as they are._ It keeps the useful helpers, keeps their tests, and lets `utils` be
the foundation. Alternatives for the name: `coola.helpers`, `coola.extras`. Alternative for the
shape: group the modules by theme (`structures`: Bloom filter, LRU, batching; `text`: string, secret,
text_diff; `system`: git, env_vars, file_size, timing), which is a better public API but a rewrite
of the paths and of the tests, not a rename. Not recommended now; a later step can do it once the
move is merged.

**Decided:** follow the recommendation.

**D1b. A `coola.formatting` sub-package?** _Recommended: yes, named `coola.formatting`._ The
functions that format are in four modules of `utils` and the formatting is the base of `display`
(`InlineDisplayMixin` builds its `repr` with `str_mapping` and `repr_indent`) and of the error
messages of `registry`. One package makes them discoverable and lets `utils` stay the foundation.
Alternatives: `coola.format` (shorter, shadows the builtin when imported as `from coola import
format`); `coola.text` (the byte and time functions are not text); fold it into `coola.display`
(`display` is the user-facing mixin; `registry` would then depend on `display`, which the layering
guide does not allow today). The sequence and mapping functions stay in one module (`structure`)
because they share `str_indent`.

**Decided:** yes, `coola.formatting`.

**D2. Delete `password`, `recursion` and `working_directory`?** _Recommended: yes._ They were the
only three items proposed for removal. `password` handles credentials (the risk of a bug is a security issue, the
benefit to `coola` is nil), `working_directory` is `contextlib.chdir`, and `recursion` explains an
error that no function raises since the explicit stacks. Alternative: keep `password` in `tools`;
it is tested (233 lines), and the risk is small if nobody calls it, but it is code that a
comparison library gets asked about in a security review.

**Decided:** remove `working_directory` only; keep `password` and `recursion` (moved to `tools`) for now and re-examine them later.

**D3. Move the default registries and the plugin loader to `coola.registry`?** _Recommended: yes._
It is registry code (it copies registries, it loads entry points into them), and it is what
`LazySingleton` should not know. The cost is a changed import in seven packages, which `lint-imports`
checks. Alternative: split `singleton.py` in `singleton.py`, `default_registry.py` and `plugins.py`
inside `utils`; cheaper, but `utils` would keep a concept it should not know.

**Decided:** follow the recommendation.

**D3b. One `optional.py` or one module per library?** _Recommended: one module._ The 11 modules differ
by a string; a single module makes the repetition visible and a parametrized test covers all of them.
The cost is 21 patches in tests and the `add_backend.md` recipe. Alternative: keep one module per
library with two functions each (about 35 lines): the same patch paths, the same recipe, 400 lines
instead of 330 and 11 test files instead of one.

**Decided:** follow the recommendation.

**D4. Enforce the role of `utils` with a test?** _Recommended: yes, as an integration test._ A rule
that rests on discipline is the rule that produced : 16 modules were added to `utils` with their
tests, each for a reason that was true on the day. The test is 30 lines and fails when a module of
`utils` is imported by no other package (it then moves to `tools` or `formatting`). It checks
modules, not names, so nothing useful is deleted. Alternative: only the sentence in the guide and
`CLAUDE.md`.

**Decided:** not now. The `ast` census test is not added; the role of `utils` is kept by review and by the `lint-imports` contracts.

**D5. Change the hash of a huge negative integer?** _Recommended: yes._ `0x-1000…` is not a
representation of the number, and a hash is only stable inside a version of `coola`
(`hashing` states that the hashes may change between major versions). Only a negative integer with
more than 4,300 digits is affected. Alternative: keep the bug and document it; not recommended
because the fix is the only way to keep `int_repr` equal to `hex(value)` for the negative case.

**Decided:** follow the recommendation.

**D6. Replace `import_object` by `pkgutil.resolve_name`?** _Recommended: yes._ Same format
(`"pkg.mod:name"`), same errors, and no second function named like the one of `coola.factory`. The
only difference is `"pkg.mod:"`, which returns the module instead of raising; nobody passes it.

**Decided:** follow the recommendation.

**D7. Make `_reset_after_fork` public?** _Recommended: no._ It is a hook for the fork handler, not
part of the API of a registry or of `LazySingleton`. The protocol (`ForkAware`) says so. A callback
passed to `track_fork(obj, callback)` would need a `WeakMethod` for the same effect, with more code.

**Decided:** follow the recommendation.

## Not covered

- **The `fallback` idiom** (`if TYPE_CHECKING or is_numpy_available(): import numpy as np else: from
coola.utils.fallback.numpy import numpy as np`) appears in 39 files. It makes `isinstance(x,
np.ndarray)` false when NumPy is missing. A different mechanism (a lazy proxy, a typed stub) is a
  separate design; this one only makes the fakes cheaper to write.
- **`coola.testing`** (the `x_available` marks, the fixtures) is only a caller here. The marks keep
  their one-row-per-dependency table.
- **`coola.display`** is only a caller here. Removing `timing` removes its dependency on `utils` for
  good, but `display` keeps importing `format`, `mapping` and `imports`.
- **Windows**: `os.register_at_fork` is absent there, so `track_fork` is a no-op (documented); the
  fork behaviour was not run on Windows.
- **Performance**: no helper of `utils` is on a hot path except `track_fork` (called once per object)
  and `is_<pkg>_available` (cached); no benchmark gate is needed for this item.
- **Import time of `importlib.metadata`** (7.5 ms) loaded by `singleton.py` for the plugin loader:
  it leaves `utils` with the loader in and is then paid by `coola.registry`, not by
  `coola.utils`.
