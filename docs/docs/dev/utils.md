# Architecture of `coola.utils`

:book: This page describes the design of the `coola.utils` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/utils.md#properties); the API is in the [reference](../refs/utils.md).

## Constraints

1. **Simple and easy to maintain.** `coola.utils` is 4,002 lines in 23 modules and two subpackages
   (`imports` 957, `fallback` 396), tested by 5,561 lines. It mixes the foundation of `coola`
   (imports, singletons, introspection) with standalone helpers (Bloom filter, LRU cache, git,
   dates) that no other package imports. The package removes the repetition (the same four functions
   were written eleven times) and deletes only what is risky or obsolete. The useful helpers are
   **kept**, with their tests.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias, no
   deprecation period (a moved module is imported from its new path). The release
   notes and the migration page (`docs/docs/migration/v1.1_to_v2.0.md`) list the changes.
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
coola/utils/                      the helpers, about 4,000 lines
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

coola/formatting/                 turning values into text, about 830 lines
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
for a message, a `repr` or a report is there (before: they were spread over `utils.format`,
`utils.string`, `utils.secret` and `utils.text_diff`, and `display` and `registry` imported the
first). It may import `utils` (`sort_by_keys`) and `validation`, and no other package. `display`,
`registry` and the feature packages import from it. The name is decision D1b: `coola.format` would
shadow the builtin in `from coola import format`.

The modules that no other package imports (`array`, `batching`, `bloom_filter`, `date`, `env_vars`,
`file_size`, `git`, `logging`, `lru`, `password`, `recursion`, `timing`) stay public and documented
in the same page as the core helpers (`docs/docs/uguide/utils.md`, `refs/utils.md`). They were moved
to a `coola.tools` package (decision D1) and then back into `coola.utils`, so there is one place for
the helpers and one import path per function.

Moved elsewhere: `coola/registry/default.py` (`make_default_registry_singleton`,
`reset_default_registries`, `temporary_default_registries`), `coola/registry/plugins.py`
(`load_registry_plugins`, `get_loaded_plugins`, `LoadedPlugin`, the two environment variable names)
and `coola/nested/conversion.py` (`to_jsonable`, which only `nested` calls and whose docstring sends
the reader to `nested`). Deleted: `get_fully_qualified_name` (replaced by
`get_qualified_class_name`), `working_directory` (`contextlib.chdir`) and `deprecate_positional_options`
(no caller).

### The modules

"Callers" are the packages of `src/coola` that import the module (outside `utils`). "Test lines" are
the lines of `tests/unit/utils/test_<module>.py`.

| Module          | Lines (test lines) | Callers                                                                                     | Role                                                                                                                                                                                |
| --------------- | ------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `array`         | 58 (81)            | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `batching`      | 51 (101)           | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `bloom_filter`  | 228 (174)          | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `date`          | 99 (139)           | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `env_vars`      | 192 (518)          | none                                                                                        | Standalone helper. `temp_env_vars` changes `os.environ` and says so (not thread-safe); it stays because it is tested, documented and `monkeypatch` is not available outside pytest. |
| `file_size`     | 55 (74)            | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `fork`          | 76 (50)            | `identifier`, `registry`                                                                    | Core. `track_fork` resets the lock of an object in a forked child.                                                                                                                  |
| `git`           | 153 (271)          | none                                                                                        | Standalone helper. The subprocess call is hardened (resolved executable, no shell, documented warning).                                                                             |
| `integer`       | 49 (68)            | `hashing`                                                                                   | Core. `int_repr` gives `-0x…` for a huge negative integer.                                                                                                                          |
| `introspection` | 278 (431)          | `equality`, `hashing`, `io`, `random`, `recursive`                                          | Core. One qualified name: `get_qualified_class_name`.                                                                                                                               |
| `logging`       | 64 (167)           | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `lru`           | 267 (308)          | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `mapping`       | 94 (145)           | `display`, `formatting`                                                                     | Core.                                                                                                                                                                               |
| `password`      | 69 (236)           | none                                                                                        | Standalone helper (D2: kept for now). It reads and validates credentials; `coola` has no use for it, so it is the first candidate to remove if it is reported as a risk.            |
| `path`          | 174 (274)          | `io`                                                                                        | Core.                                                                                                                                                                               |
| `recursion`     | 46 (37)            | none                                                                                        | Standalone helper (D2: kept for now). `equality`, `recursive` and `iterator` use explicit stacks, so nothing in `coola` raises the `RecursionError` it explains.                    |
| `safe`          | 85 (127)           | `equality`, `summary`                                                                       | Core.                                                                                                                                                                               |
| `selection`     | 55 (67)            | `summary`                                                                                   | Core. `limit` is keyword-only and checked with `validate_ge`.                                                                                                                       |
| `singleton`     | 184 (355)          | `equality`, `hashing`, `iterator`, `random`, `recursive`, `registry`, `rounding`, `summary` | Core. `LazySingleton` only: the default-registry and plugin code is in `registry`.                                                                                                  |
| `stats`         | 65 (146)           | `reducer`                                                                                   | Core. The quantile check uses `validate_range`.                                                                                                                                     |
| `tensor`        | 123 (183)          | `reducer` (`to_tensor`), `testing` (`is_cuda_available`, `is_mps_available`)                | Standalone helper that `reducer` and `testing` also use. It imports NumPy and PyTorch when they are installed.                                                                      |
| `timing`        | 105 (181)          | none                                                                                        | Standalone helper.                                                                                                                                                                  |
| `version`       | 73 (97)            | `nested`                                                                                    | Core. `get_package_version` is the building block of `compare_version`.                                                                                                             |
| `imports`       | 957 (714)          | 66 files                                                                                    | Rewritten as `package.py`, `optional.py`, `lazy.py`.                                                                                                                                |
| `fallback`      | 396 (617)          | 46 files                                                                                    | One module per library; `factory.py` takes the package name instead of a `raise_error` callable.                                                                                    |

### `imports`: one definition of "available"

The current design copies one 100-line module per library. The four functions are the same except
for a string, and two of them are unused or harmful. The target keeps what the 66 files that import it use and
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
its name for the 66 files that import it and for the `if TYPE_CHECKING or is_numpy_available():`
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
tested. A general-purpose helper that `coola` does not use also goes to `coola.utils`. Both are
generic, and a helper that is a step of one feature lives in that feature. A function that turns a
value into text goes to `coola.formatting`. No test enforces the role of `utils` (decision D4): the
review and the `lint-imports` contracts do.

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

Of the 82 public names of the 29 modules of the former `utils`, 49 stay in `utils` (24 core helpers
and 25 standalone helpers), 21 moved to `formatting`, 9 moved to `registry` (8) and `nested` (1), and
3 were deleted (`get_fully_qualified_name`, `working_directory`, `deprecate_positional_options`).
`coola.utils.imports` went from 60 public names to 35: `is_x_available` and `check_x` for the twelve
libraries and the torch/NumPy pair (26), the seven names of `package.py` and the two of `lazy.py`.
The 27 that went are the 12 decorators, the 11 `raise_x_missing_error`, `decorator_package_available`,
`LazyModule` and `lazy_import`, and `import_object`.

## Dependents

| Package                                                             | Change                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `display`, `registry`                                               | Import the formatting functions from `coola.formatting.structure`. None for the keyword-only options: no call site passes `sorted_keys`, `num_spaces` or `separator`, and `mapping_line_fn=str_mapping_line` passes the function. The import paths are unchanged.            |
| `equality`, `hashing`, `iterator`, `random`, `recursive`, `summary` | `from coola.registry.default import make_default_registry_singleton` (and `LazySingleton` from `coola.utils.singleton`); `plugin_group=` is already a keyword at the six call sites.                                                                                         |
| `testing`                                                           | `temporary_default_registries` from `coola.registry.default`; `is_cuda_available` and `is_mps_available` from `coola.utils.tensor` (the lazy `import_module` of the module name stays, so a test module does not import PyTorch).                                            |
| `reducer`                                                           | `to_tensor` from `coola.utils.tensor`; `reducer/utils.py` and `rounding/interface.py` call `pkgutil.resolve_name("pkg.mod:name")`; `registry/type.py` too.                                                                                                                   |
| `recursive`                                                         | The rebuild error uses `get_qualified_class_name(type(data))`.                                                                                                                                                                                                               |
| `nested`                                                            | `nested/conversion.py` owns the leaf converter that was `utils.conversion.to_jsonable`.                                                                                                                                                                                      |
| `hashing`                                                           | `int_repr` returns `-0x…` for a negative integer with more than 4,300 digits: the hash of such an integer changes.                                                                                                                                                           |
| `io`, `identifier`, `factory`, `equality`, `random`, `summary`      | No change: `add_uuid_suffix`, `sanitize_path`, `track_fork`, `is_module_imported`, `supports_methods`, `has_generated_dataclass_eq`, `is_same_type`, `safe_str`, `smallest_items` keep their names. The two calls of `smallest_items` in `summary` pass `limit=` by keyword. |
| 15 test files outside `tests/unit/utils`                            | 21 `patch("coola.utils.imports.<pkg>.is_<pkg>_available", ...)` become `patch("coola.utils.imports.optional.is_<pkg>_available", ...)`. The tests of the modules that moved to `formatting`, `registry` and `nested` move with their code.                                   |

## Decisions for the maintainers

Status: all decisions are taken. The recommendation was followed everywhere except two: D2 (only
`working_directory` is removed; `password` and `recursion` are kept) and D4 (no census
test for now). D1 was reversed afterwards (see below).

**D1. Name and shape of the new sub-package.** _Recommended: `coola.tools`, one module per theme,
the modules moved as they are._ It keeps the useful helpers, keeps their tests, and lets `utils` be
the foundation. Alternatives for the name: `coola.helpers`, `coola.extras`. Alternative for the
shape: group the modules by theme (`structures`: Bloom filter, LRU, batching; `text`: string, secret,
text_diff; `system`: git, env_vars, file_size, timing), which is a better public API but a rewrite
of the paths and of the tests, not a rename. Not recommended now; a later step can do it once the
move is merged.

**Decided:** follow the recommendation. `coola.tools` was created (#2569), then **moved back into
`coola.utils`** (#2579): two packages of helpers split the documentation and the import paths for no
benefit, so the standalone helpers are modules of `coola.utils` again, in the same user guide page.

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
error that no function raises since the explicit stacks. Alternative: keep `password`;
it is tested (233 lines), and the risk is small if nobody calls it, but it is code that a
comparison library gets asked about in a security review.

**Decided:** remove `working_directory` only; keep `password` and `recursion` for now and re-examine them later.

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
`utils` is imported by no other package (it then moves to a package that uses it, or is deleted). It checks
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
coola.utils.fallback.numpy import numpy as np`) appears in 46 files. It makes `isinstance(x,
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
