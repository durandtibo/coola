# Architecture of `coola.utils`

:book: This page describes the design of the `coola.utils` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/utils.md#properties); the API is in the [reference](../refs/utils.md).

## Constraints

1. **Simple and easy to maintain.** `coola.utils` mixes the foundation of `coola` (imports,
   singletons, introspection) with standalone helpers (Bloom filter, LRU cache, git, dates) that no
   other package imports. Repetition is removed (one definition of "available" for every library, one
   message for a missing package), and the useful helpers are kept, with their tests.
2. **Moved modules are imported from their new path.** There is no alias and no deprecation period.
3. **The rules of the repository apply to the helpers too.** Options are keyword-only, an argument
   check goes through `coola.validation`, a shared helper is public and tested, a lazily built
   process-wide object is a `LazySingleton`.
4. **A leaf of the layering.** `utils` is below every other package. It may import the standard
   library and `coola.validation`, nothing else (the `lint-imports` contract "Utils must import
   validation only"). `registry` depends on `utils`, never the opposite. The `formatting` package
   sits above it (it may import `utils` and `validation`).
5. **Optional libraries stay lazy.** Importing `coola.utils` or `coola.utils.imports` imports no
   optional library, and the availability helpers are the one mechanism the other packages use to
   ask whether a library is there.

## Architecture

### Layout

```
coola/utils/                      the helpers
├── __init__.py                   docstring only, no re-export
├── array.py                      to_array
├── batching.py                   batchify
├── bloom_filter.py               BloomFilter
├── date.py                       DEFAULT_DATE_FORMAT, get_today_date, to_date, to_datetime
├── env_vars.py                   check_env_vars, get_required_env_var, temp_env_vars
├── file_size.py                  get_file_size
├── fork.py                       track_fork, ForkAware
├── git.py                        run_git_command, get_current_branch, get_current_commit_hash,
│                                 is_git_repo
├── integer.py                    int_repr
├── introspection.py              get_qualified_class_name, get_type_constructor, is_same_type,
│                                 supports_methods, has_generated_dataclass_eq,
│                                 get_all_child_classes, is_lambda_function, check_not_lambda_function
├── logging.py                    CaptureHandler
├── lru.py                        LRUCache
├── mapping.py                    sort_by_keys, sort_by_values, get_first_value
├── password.py                   get_password
├── path.py                       add_uuid_suffix, sanitize_path, find_root_package_parent
├── recursion.py                  raise_recursion_limit_error
├── safe.py                       safe_repr, safe_str
├── selection.py                  smallest_items
├── singleton.py                  LazySingleton
├── stats.py                      quantile
├── tensor.py                     to_tensor, is_cuda_available, is_mps_available,
│                                 get_available_devices
├── timing.py                     TimingResult, timeblock
├── version.py                    compare_version, get_package_version
├── imports/
│   ├── __init__.py               re-exports the public names
│   ├── package.py                MissingPackageError, availability_cache, clear_availability_caches,
│   │                             package_available, module_available, check_package,
│   │                             raise_package_missing_error
│   ├── optional.py               is_<pkg>_available and check_<pkg> for the 12 libraries, and
│   │                             is_torch_numpy_available / check_torch_numpy
│   └── lazy.py                   lazy_exports, is_module_imported
└── fallback/                     one fake module per library; factory.py takes the package name

coola/formatting/                 turning values into text
├── __init__.py                   docstring only
├── structure.py                  str_indent, repr_indent, str_mapping, repr_mapping,
│                                 str_mapping_line, repr_mapping_line, str_sequence, repr_sequence,
│                                 str_sequence_line, repr_sequence_line
├── quantity.py                   str_human_byte_size, find_best_byte_unit, str_time_human, make_bar
├── text.py                       char_diff_summary, count_lines, remove_empty_lines, slugify,
│                                 truncate_str
├── secret.py                     mask_secret
└── diff.py                       unified_diff
```

`formatting` is a core package next to `utils`: every function that turns a value into a string
for a message, a `repr` or a report is there. It may import `utils` (`sort_by_keys`) and `validation`,
and no other package. `display`, `registry` and the feature packages import from it. The name is
`formatting` because `coola.format` would shadow the builtin in `from coola import format`; a
`coola.text` would be wrong because the byte and time functions are not text, and folding it into
`coola.display` would make `registry` depend on `display`, which the layering does not allow. The
sequence and mapping functions stay in one module (`structure`) because they share `str_indent`.

The modules that no other package imports (`array`, `batching`, `bloom_filter`, `date`, `env_vars`,
`file_size`, `git`, `logging`, `lru`, `password`, `recursion`, `timing`) are public and documented in
the same page as the core helpers (`docs/docs/uguide/utils.md`, `refs/utils.md`): there is one place for
the helpers and one import path per function. A package of its own for them (`coola.tools`) would split
the documentation and the import paths for no benefit; grouping them by theme (`structures`, `text`,
`system`) would be a better public API but a rewrite of the paths and of the tests.

The default-registry code (`make_default_registry_singleton`, `reset_default_registries`,
`temporary_default_registries`) and the plugin loader (`load_registry_plugins`, `get_loaded_plugins`,
`LoadedPlugin`, the two environment variable names) are in `coola.registry` (`default.py`,
`plugins.py`), because they know what a registry is (they call `copy`, they load entry points into
it). The conversion to JSON data is `coola.jsonable`.

### The modules

"Callers" are the packages of `src/coola` that import the module (outside `utils`).

| Module          | Callers                                                                                     | Role                                                                                                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `array`         | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `batching`      | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `bloom_filter`  | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `date`          | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `env_vars`      | none                                                                                        | Standalone helper. `temp_env_vars` changes `os.environ` and says so (not thread-safe); it stays because it is tested, documented and `monkeypatch` is not available outside pytest. |
| `file_size`     | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `fork`          | `identifier`, `registry`                                                                    | Core. `track_fork` resets the lock of an object in a forked child.                                                                                                                  |
| `git`           | none                                                                                        | Standalone helper. The subprocess call is hardened (resolved executable, no shell, documented warning).                                                                             |
| `integer`       | `hashing`                                                                                   | Core. `int_repr` gives `-0x…` for a huge negative integer.                                                                                                                          |
| `introspection` | `equality`, `hashing`, `io`, `random`, `recursive`                                          | Core. One qualified name: `get_qualified_class_name`.                                                                                                                               |
| `logging`       | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `lru`           | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `mapping`       | `display`, `formatting`                                                                     | Core.                                                                                                                                                                               |
| `password`      | none                                                                                        | Standalone helper. It reads and validates credentials; `coola` has no use for it, so it is the first candidate to remove if it is reported as a risk.                               |
| `path`          | `io`                                                                                        | Core.                                                                                                                                                                               |
| `recursion`     | none                                                                                        | Standalone helper. `equality`, `recursive` and `iterator` use explicit stacks, so nothing in `coola` raises the `RecursionError` it explains.                                       |
| `safe`          | `equality`, `summary`                                                                       | Core.                                                                                                                                                                               |
| `selection`     | `summary`                                                                                   | Core. `limit` is keyword-only and checked with `validate_ge`.                                                                                                                       |
| `singleton`     | `equality`, `hashing`, `iterator`, `random`, `recursive`, `registry`, `rounding`, `summary` | Core. `LazySingleton` only: the default-registry and plugin code is in `registry`.                                                                                                  |
| `stats`         | `reducer`                                                                                   | Core. The quantile check uses `validate_range`.                                                                                                                                     |
| `tensor`        | `reducer` (`to_tensor`), `testing` (`is_cuda_available`, `is_mps_available`)                | Standalone helper that `reducer` and `testing` also use. It imports NumPy and PyTorch when they are installed.                                                                      |
| `timing`        | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `version`       | `nested`                                                                                    | Core. `get_package_version` is the building block of `compare_version`.                                                                                                             |
| `imports`       | 66 files                                                                                    | `package.py`, `optional.py`, `lazy.py`.                                                                                                                                             |
| `fallback`      | 46 files                                                                                    | One module per library; `factory.py` takes the package name instead of a `raise_error` callable.                                                                                    |

### `imports`: one definition of "available"

One module names every library (`optional.py`): the functions of each library are the same except
for a string, so a library is two functions with two lines each instead of a module copied from
another library. A single module makes the repetition visible, and a parametrized test covers all of
them. One module per library (about 35 lines each) would keep the same patch paths and the same
recipe, with more lines and one test file per library.

`package.py` defines the mechanism once, with one message:

```python
class MissingPackageError(ModuleNotFoundError, RuntimeError): ...


def availability_cache(fn: F) -> F: ...
def clear_availability_caches() -> None: ...
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
its name for the files that import it and for the `if TYPE_CHECKING or is_numpy_available():`
idiom, which needs a statically known name. The torch and NumPy pair keeps its compatibility check
(`torch.tensor([1.0]).numpy()`), imported inside the function.

The public surface has no decorators `x_available`, no `raise_x_missing_error` per library (the fallback
modules call `raise_package_missing_error` through `make_fake_class("numpy")`), no `LazyModule` or
`lazy_import`, and no `import_object`: a `"pkg.mod:name"` path is resolved with
`pkgutil.resolve_name`, which has the same format and the same errors and avoids a second function
named like the one of `coola.factory`. `check_package(package, *, install=None)` has a keyword-only
`install`, the pip name when it differs from the import name, and the message is the same as for every
other package.

`lazy.py` has `lazy_exports` and `is_module_imported` (used by `factory`).

Adding a library (`docs/docs/dev/add_backend.md`) is two functions in `optional.py`, their names in
`imports/__init__.py`, one fake module in `fallback/<pkg>.py` and one row in `testing/marks.py`, plus
`pyproject.toml`: about 25 added lines.

### `LazySingleton` and the default registries

`utils/singleton.py` has the primitive, with the test hooks public so that the registry code does not
need privates:

```python
class LazySingleton(Generic[T]):
    def get(self) -> T: ...
    def is_built(self) -> bool: ...
    def reset(self) -> None:  # raises RuntimeError from the factory's own thread
        ...

    def set(self, value: T) -> None:  # replaces the cached value, under the lock
        ...
```

`coola/registry/default.py` has `make_default_registry_singleton(registry_cls, register_defaults, *,
plugin_group=None)`, `reset_default_registries()` and `temporary_default_registries()`; the last one is
written with `is_built`, `get`, `set` and `reset`: on entry, `set(get().copy())` for each registry that
is built; on exit, `set(original)` for those registries and `reset()` for the ones that were built
inside the context. `coola/registry/plugins.py` has the entry-point loader, `LoadedPlugin`,
`get_loaded_plugins` and the two environment variable names (`COOLA_STRICT_PLUGINS`,
`COOLA_DISABLE_PLUGINS`). The feature packages and `coola.testing` import from `coola.registry`;
`coola.random` also depends on `coola.registry` (its registry is a `Registry`). The import of
`importlib.metadata` (7.5 ms) is paid by `coola.registry`, not by `coola.utils`. Keeping this code in
`utils` (split in `singleton.py`, `default_registry.py` and `plugins.py`) would leave `utils` with a
concept it should not know.

### Contracts of the helpers that stay

| Function                                       | Signature                                                 | Notes                                                                                          |
| ---------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `str_indent`, `repr_indent`                    | `(original, *, num_spaces=2)`                             | keyword-only; `validate_ge(num_spaces, 0, name="num_spaces")`                                  |
| `str_mapping`, `repr_mapping`                  | `(mapping, *, sorted_keys=False, num_spaces=2)`           | keyword-only                                                                                   |
| `str_mapping_line`, `repr_mapping_line`        | `(mapping, *, sorted_keys=False, separator=", ")`         | keyword-only                                                                                   |
| `smallest_items`                               | `(data: Collection, *, limit=None)`                       | a collection (it is iterated twice when the items cannot be compared); `validate_ge(limit, 0)` |
| `int_repr`                                     | `(value: int)`                                            | `-0x…` for a huge negative integer, so `int_repr` equals `hex(value)` in the negative case     |
| `quantile`                                     | `(values, quantiles)`                                     | each quantile checked by `validate_range(q, 0.0, 1.0, name="quantiles")`                       |
| `check_package`, `raise_package_missing_error` | `(package, *, install=None)`                              | one message, keyword-only `install`                                                            |
| `make_default_registry_singleton`              | `(registry_cls, register_defaults, *, plugin_group=None)` | keyword-only; moves to `coola.registry`                                                        |
| `LazySingleton.reset`                          | `() -> None`                                              | raises `RuntimeError` when called by the thread that builds the value                          |
| `compare_version`                              | `(package, op, version)`                                  | none                                                                                           |

### Rule for a new helper

A helper used by two or more packages of `coola` goes to `coola.utils` (`CLAUDE.md`), public and
tested. A general-purpose helper that `coola` does not use also goes to `coola.utils`. Both are
generic, and a helper that is a step of one feature lives in that feature. A function that turns a
value into text goes to `coola.formatting`. No test enforces the role of `utils`: the review and the
`lint-imports` contracts do. A census test (fails when a module of `utils` is imported by no other
package) would keep the rule by construction, but the rule rests on review today.

## Public API

`coola.utils`:

```python
from coola.utils.fork import ForkAware, track_fork
from coola.utils.imports import (
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
    # scipy, torch, xarray: is_<pkg>_available, check_<pkg>; and is_torch_numpy_available,
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

The standalone helpers, which no other package imports except where stated:

```python
from coola.utils.array import to_array
from coola.utils.batching import batchify
from coola.utils.bloom_filter import BloomFilter
from coola.utils.date import DEFAULT_DATE_FORMAT, get_today_date, to_date, to_datetime
from coola.utils.env_vars import check_env_vars, get_required_env_var, temp_env_vars
from coola.utils.file_size import get_file_size
from coola.utils.git import (
    get_current_branch,
    get_current_commit_hash,
    is_git_repo,
    run_git_command,
)
from coola.utils.logging import CaptureHandler
from coola.utils.lru import LRUCache
from coola.utils.password import get_password
from coola.utils.recursion import raise_recursion_limit_error
from coola.utils.tensor import (  # `reducer` and `testing` import from it
    get_available_devices,
    is_cuda_available,
    is_mps_available,
    to_tensor,
)
from coola.utils.timing import TimingResult, timeblock
```

## Dependents

| Package                                                             | Use                                                                                                                                                                                                |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `display`, `registry`                                               | The formatting functions from `coola.formatting.structure`.                                                                                                                                        |
| `equality`, `hashing`, `iterator`, `random`, `recursive`, `summary` | `make_default_registry_singleton` from `coola.registry.default`, and `LazySingleton` from `coola.utils.singleton`.                                                                                 |
| `testing`                                                           | `temporary_default_registries` from `coola.registry.default`; `is_cuda_available` and `is_mps_available` from `coola.utils.tensor` (imported lazily, so a test module does not import PyTorch).    |
| `reducer`                                                           | `to_tensor` from `coola.utils.tensor`; `reducer/utils.py` and `rounding/interface.py` resolve `"pkg.mod:name"` paths with `pkgutil.resolve_name`, as `registry/type.py` does.                      |
| `recursive`                                                         | The rebuild error uses `get_qualified_class_name(type(data))`.                                                                                                                                     |
| `hashing`                                                           | `int_repr` returns `-0x…` for a negative integer with more than 4,300 digits.                                                                                                                      |
| `io`, `identifier`, `factory`, `equality`, `random`, `summary`      | `add_uuid_suffix`, `sanitize_path`, `track_fork`, `is_module_imported`, `supports_methods`, `has_generated_dataclass_eq`, `is_same_type`, `safe_str`, `smallest_items` (with `limit=` by keyword). |

## Design choices and trade-offs

**Standalone helpers stay in `utils`.** Deleting a helper that no package uses is risky for users and
cheap to keep: each has tests and a documented role. Two helpers carry a caveat: `password` handles
credentials (a bug would be a security issue, and `coola` has no use for it), and `recursion` explains
an error that no function raises since the explicit stacks. They are kept for now and are the first
candidates for removal. `working_directory` is not provided (`contextlib.chdir`), nor
`deprecate_positional_options`.

**The default registries and the plugin loader live in `coola.registry`.** It is registry code (it
copies registries, it loads entry points into them), and it is what `LazySingleton` should not know.
The cost is an import in each feature package, which `lint-imports` checks.

**A huge negative integer is hashed as `-0x…`.** `0x-1000…` is not a representation of the number. Only
a negative integer with more than 4,300 digits is affected. A hash is only stable inside a version of
`coola`.

**`_reset_after_fork` is not public.** It is a hook for the fork handler, not part of the API of a
registry or of `LazySingleton`; the protocol (`ForkAware`) says so. A callback passed to
`track_fork(obj, callback)` would need a `WeakMethod` for the same effect, with more code.

## Not covered

- **The `fallback` idiom** (`if TYPE_CHECKING or is_numpy_available(): import numpy as np else: from
coola.utils.fallback.numpy import numpy as np`) appears in 46 files. It makes `isinstance(x,
np.ndarray)` false when NumPy is missing. A different mechanism (a lazy proxy, a typed stub) is a
  separate design; this one only makes the fakes cheaper to write.
- **`coola.testing`** (the `x_available` marks, the fixtures) is only a caller here. The marks keep
  their one-row-per-dependency table.
- **`coola.display`** is only a caller here. Removing `timing` removes its dependency on `utils` for
  good, but `display` keeps importing `format`, `mapping` and `imports`.
- **Windows**: `os.register_at_fork` is absent there, so `track_fork` is a no-op (documented); the
  fork behaviour is not tested on Windows.
- **Performance**: no helper of `utils` is on a hot path except `track_fork` (called once per object)
  and `is_<pkg>_available` (cached); no benchmark gate is needed for this item.
