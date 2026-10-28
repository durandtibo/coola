# `coola.utils`: target state and the changes to reach it

This document describes the design that `coola.utils` should have, then the changes to make to the
current implementation (`main`, `dfb12b36`) to get there. The design follows the rules of the user
guide (`docs/docs/uguide/utils.md`, **Properties**), the architecture guide
(`docs/docs/dev/architecture.md`: package layering, lazy imports, optional dependencies, strict
contracts) and the rules of `CLAUDE.md` (keyword-only options, validation through
`coola.validation`, public shared helpers, `LazySingleton`, Google docstrings with a doctest). It stays
aligned with the sibling core packages `validation` and `registry`. Its aim is a small package that is
easy to maintain: **a leaf layer in which every public name has a caller, one definition of "an
optional package is available", and one place for each kind of state**. Breaking changes are
accepted (the version is `2.0.0a0`), and the future state is more important than the past one.

Evidence comes from probes, from a census of the imports of `src` and `tests` (an `ast` walk), and
from a small prototype run on `main` (Python 3.14.8, scripts kept outside the repository). Items marked
**[verified]** were reproduced; **[reading]** comes from the code. Nothing in `src`, `tests` or `docs`
was changed.

## Constraints

1. **Simple and easy to maintain.** `coola.utils` is 6,048 lines in 29 modules and two subpackages
   (`imports` 1,742, `fallback` 421, the 29 modules 3,885), tested by 9,196 lines. The target removes
   the code that no other package calls, and the repetition (the same four functions written eleven
   times) instead of adding code next to it. About 1,950 lines remain.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias, no
   deprecation period (the `deprecation` helper itself has no caller and is deleted). The release
   notes and the migration page (`docs/docs/migration/v1.1_to_v2.0.md`) list the changes.
3. **The rules of the repository apply to the helpers too.** Options are keyword-only, an argument
   check goes through `coola.validation`, a shared helper is public and tested, a lazily built
   process-wide object is a `LazySingleton`. Today 10 of the surviving functions break one of the
   first two rules (F9).
4. **A leaf of the layering.** `utils` is below every other package. It may import the standard
   library and `coola.validation`, nothing else (F2). `registry` depends on `utils`, never the
   opposite.
5. **Optional libraries stay lazy.** Importing `coola.utils` or `coola.utils.imports` imports no
   optional library, and the availability helpers are the one mechanism the other packages use to
   ask whether a library is there.

## 1. Properties of the sub-package

These are the rules of the target. Today the guide lists four loose properties (optional
dependencies, cached checks, focused submodules, informative errors) and a design-principles list
("Modular", "Efficient", "Flexible") that does not decide anything. The table below replaces both,
and each row has an owner: a test, a lint contract or a rule of the repository.

| Property                         | Statement                                                                                                                                                                                                                                                                                                                                                                                 | Enforced by                                                 |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Leaf layer                       | `coola.utils` imports the standard library and `coola.validation` only. It imports no feature package, no `registry`, no `display`. `coola.validation` does not import `coola.utils`.                                                                                                                                                                                                     | `lint-imports` (two new contracts, step 2)                  |
| Earned place                     | Every public name of `coola.utils` is used by another `coola` package, or is a building block of a name that is. A helper without a caller is deleted, not "kept because it is small". A helper with one caller that is generic stays; a helper that exists only as a step of one feature moves to that feature.                                                                          | a test that walks `src` with `ast` (decision D4)            |
| Generic                          | A helper knows nothing of the domain of a package: not what a registry is, not what a plugin group is, not what a JSON document of `nested` is. Domain code lives with its domain.                                                                                                                                                                                                        | review, the table of section 3.2                            |
| One name per concept             | One concept has one function and one message. No two functions share a name with different formats (`import_object`), no two functions return the qualified name of a class, no two messages report the same missing package.                                                                                                                                                             | review, the table of section 3.2                            |
| Optional packages stay optional  | `is_<pkg>_available()` is the single definition of "available" (the package can be imported); it is cached and `clear_availability_caches()` resets it. `check_<pkg>()` raises `MissingPackageError`, with one message for all packages. `package_available(name)` looks a package up without importing it. The fake modules of `fallback` keep annotations and `isinstance` working.     | `tests/unit/utils/imports`, `tests/integration/utils`       |
| Import-light                     | `import coola.utils.<module>` imports no optional library, calls no check and does no work, except `os.register_at_fork` in `fork`. The only module that uses the `if TYPE_CHECKING or is_x_available()` idiom is `version` (`packaging`).                                                                                                                                                | `tests/integration/test_lazy_imports.py` (extended)         |
| Strict contracts                 | An argument is checked by `coola.validation` (a wrong type is a `TypeError`, a wrong value a `ValueError`, with the message of the validator). Options are keyword-only. No helper hides an error: nothing returns `None` instead of raising, nothing swallows an exception unless reporting it as a value is the job of the function (`safe_str`, `safe_repr`, the availability checks). | `ruff`, review, unit tests of the messages                  |
| Stateless, with named exceptions | A helper keeps no state, with three exceptions that are documented and tested: the availability caches, the registry of the objects that reset after a fork (`track_fork`) and the `LazySingleton` instances. No helper changes `os.environ`, the working directory or a logger.                                                                                                          | review, `tests/integration/utils`                           |
| Thread-safe, fork-safe           | A `LazySingleton` builds its value once under a lock, detects a request from its own factory, and rebuilds its lock in a forked child. The availability caches are `functools.cache` (a duplicate computation is harmless).                                                                                                                                                               | `tests/integration/utils/test_singleton.py`, `test_fork.py` |
| Explicit modules                 | There is no re-export at `coola.utils` (`__all__` is empty, as today): a name is imported from the module that defines it (`coola.utils.format`), except `coola.utils.imports`, which re-exports its public names.                                                                                                                                                                        | review                                                      |
| Public, documented, tested       | A name used by another module has no leading underscore. Each public function has a Google docstring with a `pycon` example that runs as a doctest, one test module per source module, 100 % coverage, and a line in `docs/docs/refs/utils.md` (all the modules, not half of them).                                                                                                       | `xdoctest`, `pytest --cov`, `mkdocs build --strict`         |

Not guaranteed, and documented as such:

- **The availability of a package cannot change during the life of the process** (it is cached on
  the first call). A test that changes `sys.modules` calls `clear_availability_caches()`.
- **`is_<pkg>_available()` imports the package** the first time (a package that is installed but
  broken is reported as not available, with a logged warning). Use `package_available(name)` to ask
  whether a package is installed without importing it.
- **`LazySingleton` is not a registry of its values.** `reset()` forgets the value, and a reference
  obtained before is not changed.
- **The order of `smallest_items` when the items cannot be compared** is the iteration order of the
  collection.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Status         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| F1  | **There is no rule for what belongs to `utils`.** Of the 82 public names of the 29 modules, 29 are used by another package, 1 only by `utils`, and **52 by nobody**. Sixteen whole modules have no caller: `array`, `batching`, `bloom_filter`, `date`, `deprecation`, `env_vars`, `file_size`, `git`, `logging`, `lru`, `password`, `recursion`, `secret`, `string`, `text_diff`, `timing` (1,711 of the 3,885 lines, 2,789 lines of unit tests). The guide calls them "general helpers, kept because they are small and tested". In total they are not small, and some carry a risk that the rest of the package does not have: `git` runs a subprocess, `password` reads the terminal, `temp_env_vars` and `working_directory` change process-wide state. | **[verified]** |
| F2  | **`utils` and `display` import each other.** `utils/timing.py` imports `coola.display.InlineDisplayMixin` and `display` imports `utils.format`, `utils.imports`, `utils.mapping`. The architecture guide accepts it ("kept in the same group") and `lint-imports` cannot forbid it. `timing` is called by nobody, and it is the only module of `utils` that imports another `coola` package other than `validation`.                                                                                                                                                                                                                                                                                                                                         | **[verified]** |
| F3  | **`singleton.py` has four responsibilities in 428 lines.** `LazySingleton` (the primitive, used by six packages); `make_default_registry_singleton`, which builds and populates a default registry; the plugin loader (`load_registry_plugins`, entry points, two environment variables, `ExceptionGroup`, `get_loaded_plugins`); and `reset_default_registries` / `temporary_default_registries`, which call the private methods `_replace_with_copy` and `_restore` of `LazySingleton` and call `.copy()` on a registry. Three of the four are about registries, which `utils` should not know.                                                                                                                                                            | **[reading]**  |
| F4  | **`imports` is 1,742 lines for one idea.** Eleven libraries each have a 96 to 102 line module with the same four functions (`check_x`, `is_x_available`, `x_available`, `raise_x_missing_error`), a fallback module, and 44 names in `__init__`. The decorator `x_available` has **no caller** (the `x_available` marks of the tests are those of `coola.testing`) and it returns `None` silently when the package is missing, which breaks "an error is not hidden"; the eleven `raise_x_missing_error` are called by the fallback modules only; `LazyModule` and `lazy_import` have no caller (and keep a licence file, `licenses/LICENSE_lightning_utilities`).                                                                                           | **[verified]** |
| F5  | **Two messages for the same error.** `check_package("zzz", "pip install zzz")` and `raise_package_missing_error("jax", "jax")` do not say the same thing (one asks for a command, the other builds `pip install` and `uv pip install`). `check_package` tests `package_available` (no import), while `check_numpy` tests `is_numpy_available` (imports): the two checks of the same family do not mean the same.                                                                                                                                                                                                                                                                                                                                             | **[verified]** |
| F6  | **Two `import_object`.** `coola.utils.imports.import_object("pkg.mod:name")` and `coola.factory.import_object("pkg.mod.name")` have the same name and a different format; each docstring warns about the other. The standard library already parses the colon form: `pkgutil.resolve_name("collections:OrderedDict")`. It raises `ValueError` for `""` and `":x"`, `ModuleNotFoundError` and `AttributeError` for a missing target, as `import_object` does. The three callers pass constants.                                                                                                                                                                                                                                                               | **[verified]** |
| F7  | **Two qualified names.** `get_fully_qualified_name(K)` returns `K` and `get_qualified_class_name(K)` returns `__main__.K` for a class defined in `__main__`; for `int` they return `builtins.int` and `int`. The first has one caller (the rebuild error of `recursive`, always on a type), the second has one caller (`hashing`).                                                                                                                                                                                                                                                                                                                                                                                                                           | **[verified]** |
| F8  | **Three defects in code that stays.** `int_repr(-(1 << 20000))` returns `0x-1000…`: the sign is inside the prefix (the positive case is `0x1000…`). `smallest_items(iter([3, "a", 1]))` returns `[]`: `sorted` consumes the iterator, then the fallback `islice` finds it empty. `LazySingleton(lambda: s.reset())` deadlocks: `reset` takes the lock that `get` holds (the thread that builds the value).                                                                                                                                                                                                                                                                                                                                                   | **[verified]** |
| F9  | **Contracts that break the rules of the repository.** Positional options: `str_mapping`, `repr_mapping` (`sorted_keys`, `num_spaces`), `str_mapping_line`, `repr_mapping_line` (`sorted_keys`, `separator`), `str_sequence`, `repr_sequence`, `str_indent`, `repr_indent` (`num_spaces`), `make_default_registry_singleton` (`plugin_group`), `smallest_items` (`limit`), `check_package` (`command`). A raw `raise ValueError` for a range check: `str_indent` (negative `num_spaces`), `quantile` (a quantile outside `[0, 1]`), `make_bar`, `str_human_byte_size`, `BloomFilter`, `truncate_str`. Only the first group matters after the deletions of F1.                                                                                                 | **[reading]**  |
| F10 | **The guide contradicts the code.** `with numpy_available(): ...` is documented as a context manager; it is a decorator and the call raises `TypeError: numpy_available() missing 1 required positional argument: 'fn'`. The "core helpers used by the other packages" include `array`, `string`, `env_vars` and `logging`, which nobody calls. `docs/docs/refs/utils.md` lists 14 of the 29 modules (missing: `batching`, `conversion`, `deprecation`, `file_size`, `fork`, `git`, `integer`, `logging`, `lru`, `recursion`, `safe`, `selection`, `singleton`, `text_diff`, `timing`) and not `fallback`.                                                                                                                                                   | **[verified]** |
| F11 | **`array` and `tensor` import NumPy and PyTorch at module level** (guarded, but they import the library when it is installed), for three uses: `to_array` has no caller; `to_tensor` has one, three lines for a NumPy array in `reducer/torch.py`; `is_cuda_available` and `is_mps_available` are called by `coola.testing` only, through `import_module("coola.utils.tensor")`. `is_mps_available` allocates a tensor on the device (0.44 s in its unit test). `get_available_devices` has no caller.                                                                                                                                                                                                                                                       | **[verified]** |
| F12 | **`format.py` has 14 functions, 6 are used.** `make_bar`, `str_sequence`, `repr_sequence`, `str_sequence_line`, `repr_sequence_line`, `str_human_byte_size`, `find_best_byte_unit` have no caller, and `str_time_human` is called by `timing` only. The six that are used (`str_indent`, `repr_indent`, `str_mapping`, `repr_mapping`, `str_mapping_line`, `repr_mapping_line`) are called by `display` and by `registry`.                                                                                                                                                                                                                                                                                                                                   | **[verified]** |
| F13 | **`to_jsonable` is half of an API.** It converts one object, its docstring sends the reader to `coola.nested.convert_to_jsonable` four times (containers, dataclasses, sets, nested data), and `nested.conversion` is its only caller.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | **[reading]**  |

What is already right and must not regress: `LazySingleton` (double-checked lock, detection of a
request from its own factory, lock rebuilt after a fork); `availability_cache` /
`clear_availability_caches` (the functions to reset are the ones decorated, there is no list to
maintain); `module_available` (reports a broken installation and says why); `MissingPackageError`
(a `ModuleNotFoundError` and a `RuntimeError`); `track_fork` (a weak set, no strong reference);
`safe_str` / `safe_repr`; `lazy_exports`; the fake modules of `fallback`; the strict-plugin mode
(`COOLA_STRICT_PLUGINS`); `add_uuid_suffix` and `sanitize_path`; `compare_version`. The tests are fast
(1,277 passed and 18 skipped in 5.7 s) and cover 99 % of the package.

## 3. Architecture

### 3.1 Layout

```
coola/utils/                      about 1,950 lines (6,048 today)
├── __init__.py                   docstring only, no re-export
├── fork.py                       track_fork, ForkAware
├── format.py                     str_indent, repr_indent, str_mapping, repr_mapping,
│                                 str_mapping_line, repr_mapping_line
├── integer.py                    int_repr
├── introspection.py              get_qualified_class_name, get_type_constructor, is_same_type,
│                                 supports_methods, has_generated_dataclass_eq
├── mapping.py                    sort_by_keys
├── path.py                       add_uuid_suffix, sanitize_path
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
```

Moved out of `utils`: `coola/registry/default.py` (`make_default_registry_singleton`,
`reset_default_registries`, `temporary_default_registries`), `coola/registry/plugins.py`
(`load_registry_plugins`, `get_loaded_plugins`, `LoadedPlugin`, the two environment variable names),
`coola/nested/conversion.py` (the leaf converter `to_jsonable`, private), `coola/reducer/torch.py`
(the three lines of `to_tensor`), `coola/testing/marks.py` (`is_cuda_available`, `is_mps_available`).
Deleted: sixteen modules and the unused names of the other modules (section 3.2).

### 3.2 Disposition of every module

"Callers" are the packages of `src/coola` that import the name (outside `utils`), from the census of
section 6. "Test lines" are the lines of `tests/unit/utils/test_<module>.py`.

| Module          | Lines (test lines) | Callers today                                                                  | Target                                                                                                                                                                    |
| --------------- | ------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `array`         | 58 (81)            | none                                                                           | **Delete.** `to_array` has no caller.                                                                                                                                     |
| `batching`      | 51 (101)           | none                                                                           | **Delete.**                                                                                                                                                               |
| `bloom_filter`  | 213 (174)          | none                                                                           | **Delete.**                                                                                                                                                               |
| `conversion`    | 115 (211)          | `nested`                                                                       | **Move** to `nested/conversion.py` as a private leaf converter (F13).                                                                                                     |
| `date`          | 99 (139)           | none                                                                           | **Delete.**                                                                                                                                                               |
| `deprecation`   | 112 (97)           | none                                                                           | **Delete.** The next deprecation re-adds what it needs.                                                                                                                   |
| `env_vars`      | 188 (518)          | none                                                                           | **Delete.** (The two plugin variables are read by `plugins.py` with its own 2-line helper.)                                                                               |
| `file_size`     | 55 (74)            | none                                                                           | **Delete.**                                                                                                                                                               |
| `fork`          | 76 (50)            | `identifier`, `registry`, `singleton`                                          | **Keep.**                                                                                                                                                                 |
| `format`        | 491 (587)          | `display`, `registry`                                                          | **Trim** to the six used functions; options keyword-only; `validate_ge` for `num_spaces` (F9, F12).                                                                       |
| `git`           | 153 (271)          | none                                                                           | **Delete.**                                                                                                                                                               |
| `integer`       | 48 (64)            | `hashing`                                                                      | **Keep, fix the sign** of a huge negative integer (F8).                                                                                                                   |
| `introspection` | 322 (514)          | `equality`, `hashing`, `io`, `random`, `recursive`                             | **Trim** to five functions. Delete `get_fully_qualified_name` (F7), `get_all_child_classes`, `is_lambda_function`, `check_not_lambda_function`.                           |
| `logging`       | 50 (167)           | none (one integration test of itself)                                          | **Delete.**                                                                                                                                                               |
| `lru`           | 204 (308)          | none (the docstring says it is not used by the registries)                     | **Delete.**                                                                                                                                                               |
| `mapping`       | 94 (145)           | `display`, `format`                                                            | **Trim** to `sort_by_keys`. Delete `sort_by_values` and `get_first_value`.                                                                                                |
| `password`      | 69 (233)           | none                                                                           | **Delete.**                                                                                                                                                               |
| `path`          | 212 (396)          | `io`                                                                           | **Trim** to `add_uuid_suffix` and `sanitize_path`. `working_directory` is `contextlib.chdir`; `find_root_package_parent` has no caller.                                   |
| `recursion`     | 46 (37)            | none (the explicit stacks of `equality` and `recursive` replaced it)           | **Delete.**                                                                                                                                                               |
| `safe`          | 76 (126)           | `equality`, `summary`                                                          | **Keep.**                                                                                                                                                                 |
| `secret`        | 48 (55)            | none                                                                           | **Delete.**                                                                                                                                                               |
| `selection`     | 46 (58)            | `summary`                                                                      | **Keep, fix the iterator and make `limit` keyword-only with `validate_ge`** (F8, F9).                                                                                     |
| `singleton`     | 428 (729)          | `equality`, `hashing`, `iterator`, `random`, `recursive`, `summary`, `testing` | **Split** (F3): `LazySingleton` stays; the default-registry and plugin code moves to `registry`.                                                                          |
| `stats`         | 64 (146)           | `reducer`                                                                      | **Keep**; the quantile check uses `validate_range`.                                                                                                                       |
| `string`        | 203 (224)          | none                                                                           | **Delete.**                                                                                                                                                               |
| `tensor`        | 123 (183)          | `reducer` (`to_tensor`), `testing` (`is_cuda_available`, `is_mps_available`)   | **Delete.** `to_tensor` becomes three private lines in `reducer/torch.py`; the two device checks move to `testing/marks.py`; `get_available_devices` has no caller (F11). |
| `text_diff`     | 69 (129)           | none                                                                           | **Delete.**                                                                                                                                                               |
| `timing`        | 93 (181)           | none                                                                           | **Delete** (and with it the cycle of F2).                                                                                                                                 |
| `version`       | 73 (97)            | `nested`                                                                       | **Keep.** `get_package_version` is the building block of `compare_version`.                                                                                               |
| `imports`       | 1,742 (1,897)      | 59 files                                                                       | **Rewrite** as `package.py`, `optional.py`, `lazy.py` (section 3.3).                                                                                                      |
| `fallback`      | 421 (579)          | 39 files                                                                       | **Keep the shape**: one module per library; `factory.py` takes the package name instead of a `raise_error` callable.                                                      |

### 3.3 `imports`: one definition of "available"

The current design copies one 100-line module per library. The four functions are the same except
for a string, and two of them are unused or harmful (F4). The target keeps what the 59 files that import it use and
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

so that `check_numpy()` is by definition `check_package("numpy")` (F5), and `is_numpy_available` keeps
its name for the 59 files that import it and for the `if TYPE_CHECKING or is_numpy_available():`
idiom, which needs a statically known name. The torch and NumPy pair keeps its compatibility check
(`torch.tensor([1.0]).numpy()`), imported inside the function.

Deleted from the public surface: the eleven decorators `x_available` and `torch_numpy_available`
(no caller, silent `None`), `decorator_package_available`, the eleven `raise_x_missing_error`
(the fallback modules call `raise_package_missing_error` through `make_fake_class("numpy")`),
`LazyModule`, `lazy_import` and `import_object` (replaced by `pkgutil.resolve_name`, F6). The
`check_package` signature loses its positional `command`: the `install` keyword is the pip name, and
the message is the same as for every other package.

`lazy.py` keeps `lazy_exports` (eight packages) and `is_module_imported` (`factory`).

Adding a library (`docs/docs/dev/add_backend.md`) is two functions in `optional.py`, their names in
`imports/__init__.py`, one fake module in `fallback/<pkg>.py` and one row in `testing/marks.py`, plus
`pyproject.toml`: about 25 added lines instead of a 100-line module copied from `pandas.py`.

### 3.4 `LazySingleton` and the default registries

`utils/singleton.py` keeps the primitive and makes the two private test hooks public, so that the
code that moves out does not need the privates (F3):

```python
class LazySingleton(Generic[T]):
    def get(self) -> T: ...  # unchanged
    def is_built(self) -> bool: ...  # unchanged
    def reset(self) -> None:  # raises RuntimeError from the factory's own thread (F8)
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

### 3.5 Contracts of the helpers that stay

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

### 3.6 Rule for a new helper

A new helper goes to `coola.utils` when it is used by **two** places, as `CLAUDE.md` says. It is
generic (3.2, "Generic"), public, documented, and checked by the `ast` test of section 8 as soon as
it lands: if the second caller does not land with it, the helper stays in the package that owns the
first one.

## 4. Public API after the change

```python
from coola.utils.fork import ForkAware, track_fork
from coola.utils.format import (
    repr_indent,
    repr_mapping,
    repr_mapping_line,
    str_indent,
    str_mapping,
    str_mapping_line,
)
from coola.utils.imports import (  # same names as today, minus the removed ones
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
    get_qualified_class_name,
    get_type_constructor,
    has_generated_dataclass_eq,
    is_same_type,
    supports_methods,
)
from coola.utils.mapping import sort_by_keys
from coola.utils.path import add_uuid_suffix, sanitize_path
from coola.utils.safe import safe_repr, safe_str
from coola.utils.selection import smallest_items
from coola.utils.singleton import LazySingleton
from coola.utils.stats import quantile
from coola.utils.version import compare_version, get_package_version
```

Of the 82 public names of the 29 modules, 24 remain in `utils`, 12 move (six with a caller, six
supporting names of the plugin code) and 46 are deleted (45 without a caller, and
`get_fully_qualified_name`, merged into `get_qualified_class_name`). `coola.utils.imports` goes from
60 public names to 33: `is_x_available` and `check_x` for the eleven libraries and the torch/NumPy
pair (24), the seven names of `package.py` and the two of `lazy.py`. The 27 that go are the 12
decorators, the 11 `raise_x_missing_error`, `decorator_package_available`, `LazyModule` and
`lazy_import`, and `import_object`.

## 5. Dependents

| Package                                                             | Change                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `display`, `registry`                                               | None for the keyword-only options: no call site passes `sorted_keys`, `num_spaces` or `separator`, and `mapping_line_fn=str_mapping_line` passes the function. The import paths are unchanged.                                                                               |
| `equality`, `hashing`, `iterator`, `random`, `recursive`, `summary` | `from coola.registry.default import make_default_registry_singleton` (and `LazySingleton` from `coola.utils.singleton`); `plugin_group=` is already a keyword at the six call sites.                                                                                         |
| `testing`                                                           | `temporary_default_registries` from `coola.registry.default`; `_is_cuda_available` and `_is_mps_available` become local functions of `marks.py` (the code moves, the lazy `import_module` goes away).                                                                        |
| `reducer`                                                           | `reducer/torch.py` calls `torch.from_numpy` itself (copying a negatively strided array first); `reducer/utils.py` and `rounding/interface.py` call `pkgutil.resolve_name("pkg.mod:name")`; `registry/type.py` too.                                                           |
| `recursive`                                                         | The rebuild error uses `get_qualified_class_name(type(data))`.                                                                                                                                                                                                               |
| `nested`                                                            | `nested/conversion.py` owns the leaf converter that was `utils.conversion.to_jsonable`.                                                                                                                                                                                      |
| `hashing`                                                           | `int_repr` returns `-0x…` for a negative integer with more than 4,300 digits: the hash of such an integer changes (D5).                                                                                                                                                      |
| `io`, `identifier`, `factory`, `equality`, `random`, `summary`      | No change: `add_uuid_suffix`, `sanitize_path`, `track_fork`, `is_module_imported`, `supports_methods`, `has_generated_dataclass_eq`, `is_same_type`, `safe_str`, `smallest_items` keep their names. The two calls of `smallest_items` in `summary` pass `limit=` by keyword. |
| 15 test files outside `tests/unit/utils`                            | 21 `patch("coola.utils.imports.<pkg>.is_<pkg>_available", ...)` become `patch("coola.utils.imports.optional.is_<pkg>_available", ...)`.                                                                                                                                      |

## 6. Evidence

Probes and the census were run on `main` (`dfb12b36`). The scripts are in the scratchpad of the
session, outside the repository, and are described so they can be rewritten: the census parses
every `*.py` of `src` and `tests` with `ast`, collects the `from coola.utils.<module> import <name>`
and `import coola.utils.<module>` statements outside `coola.utils`, and compares them with the
`__all__` of each module. The two names that `coola.testing` reaches through `import_module`
(`is_cuda_available`, `is_mps_available`) were added by hand.

**Baseline.** `pytest tests/unit/utils tests/integration/utils --cov=coola.utils`: 1,277 passed, 18
skipped in 5.7 s, 99 % (one missed line, `introspection.py:248`). `lint-imports`: 2 contracts kept,
0 broken. `ruff check`, `ty check` and `xdoctest` were not run for this review (nothing was changed).

| Probe                                                                                    | Result                                                                                                                                    | Finding  |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| census: public names with / without a caller                                             | 82 names; 29 with a caller outside `utils`, 1 used only inside, **52 without**; 16 modules without any caller                             | F1       |
| imports of other `coola` packages from `utils`                                           | `coola.validation` (6 imports) and `coola.display.InlineDisplayMixin` (`timing.py` only)                                                  | F2       |
| `with numpy_available(): pass`                                                           | `TypeError: numpy_available() missing 1 required positional argument: 'fn'`                                                               | F10      |
| `check_package("zzz", "pip install zzz")` vs `raise_jax_missing_error()`                 | two different messages                                                                                                                    | F5       |
| `pkgutil.resolve_name` on `""`, `":x"`, `"a"`, `"collections:Nope"`, `"nomod:x"`         | `ValueError`, `ValueError`, `ModuleNotFoundError`, `AttributeError`, `ModuleNotFoundError`; `"collections:OrderedDict"` returns the class | F6       |
| `get_fully_qualified_name(K)` / `get_qualified_class_name(K)` for `K` in `__main__`      | `'K'` / `'__main__.K'`; for `int`: `'builtins.int'` / `'int'`                                                                             | F7       |
| `int_repr(-(1 << 20000))[:12]`                                                           | `'0x-100000000'`                                                                                                                          | F8       |
| `smallest_items(iter([3, "a", 1]))`                                                      | `[]` (the set `{3, "a", 1}` gives `[1, 'a', 3]`)                                                                                          | F8       |
| `LazySingleton(lambda: s.reset())`, `s.get()` from a thread, joined after 1 s            | the thread is still running: deadlock                                                                                                     | F8       |
| `-X importtime`: `import coola.utils.singleton, coola.utils.format, coola.utils.mapping` | no optional library in `sys.modules` (`numpy`, `torch`, `pandas`, `packaging`); `importlib.metadata` is the costly import (7.5 ms)        | property |
| `is_numpy_available()` first call                                                        | 58 ms and `numpy` is in `sys.modules`: the check imports the package (documented in the properties)                                       | property |
| `str_indent(x, -1)`, `make_bar(2.0)`, `str_human_byte_size(5, "XB")`                     | raw `ValueError` with a hand-written message                                                                                              | F9       |

**Prototype.** The `optional.py` of section 3.3 was written for two libraries (`numpy`, `torch`)
against a `package.py` of 30 lines (`MissingPackageError`, `availability_cache`, `module_available`,
`check_package`). It imports, `is_numpy_available()` and `is_torch_available()` return `True`,
`check_numpy()` passes, and the file is 45 lines with the short docstrings used for the prototype.
With the docstring convention of the repository (Returns, Raises, a doctest) the estimate is 25
lines per library, so about 330 lines for eleven libraries and the torch/NumPy pair, instead of
1,100 for the eleven modules. `ty` was not run on the prototype: the real type check happens in
step 4.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable and
puts the small defects and the deletions first, because every later step is smaller when the deleted
code is gone.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Breaking                                    |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 1    | **Fix the three defects (F8).** `int_repr` sign, `smallest_items` with an iterator (type `Collection`), `LazySingleton.reset` from the factory thread. One regression test each, each checked to fail without the fix (`git stash push src`).                                                                                                                                                                                                                                                                                           | hash of a huge negative integer (D5)        |
| 2    | **Delete what nobody calls (F1, F2, F4 in part, F11, F12).** The 16 modules, the unused names of `format`, `introspection`, `mapping`, `path`, `tensor` (with `is_cuda_available` / `is_mps_available` moved to `testing/marks.py` and `to_tensor` inlined in `reducer`), `LazyModule`, `lazy_import` and `licenses/LICENSE_lightning_utilities`. Delete their tests. Add the two `lint-imports` contracts and the `ast` census test (D4).                                                                                              | yes (names removed)                         |
| 3    | **Fix the documentation to match the code (F10).** `numpy_available` is a decorator (or is gone after step 4); the "core helpers" list; `refs/utils.md` lists every module. Documentation only, can be merged before step 2.                                                                                                                                                                                                                                                                                                            | no                                          |
| 4    | **Rewrite `imports` (F4, F5, F6).** `package.py`, `optional.py`, `lazy.py`; one message; delete the decorators, `raise_x_missing_error` and `decorator_package_available`; `make_fake_class("numpy")`; `pkgutil.resolve_name` replaces `import_object`; migrate the 21 patches; update `add_backend.md` and `testing.md`. Parametrize the tests over the 11 libraries and the pair.                                                                                                                                                     | yes (decorators, `check_package` signature) |
| 5    | **Contracts (F9).** Keyword-only options and `coola.validation` in `format`, `selection`, `stats`, `check_package`; the two calls of `smallest_items` in `summary` pass `limit=`; no call site of `format` passes an option.                                                                                                                                                                                                                                                                                                            | yes (positional options)                    |
| 6    | **Merge the qualified names (F7).** Delete `get_fully_qualified_name`; `recursive` uses `get_qualified_class_name(type(data))`.                                                                                                                                                                                                                                                                                                                                                                                                         | message of the rebuild error                |
| 7    | **Move `to_jsonable` to `nested` (F13).** Private leaf converter in `nested/conversion.py`, its tests move with it.                                                                                                                                                                                                                                                                                                                                                                                                                     | yes (name removed)                          |
| 8    | **Split `singleton` (F3).** `LazySingleton.set`; `registry/default.py` and `registry/plugins.py`; the six packages and `testing` change their import; the tests move to `tests/unit/registry`, the integration tests to `tests/integration/registry`.                                                                                                                                                                                                                                                                                   | yes (import path)                           |
| 9    | **Documentation.** Rewrite `uguide/utils.md` (the properties of section 1, the modules of section 4, no "general helpers" table), `refs/utils.md`, the layering paragraph of `architecture.md` (remove "`display` and `utils` currently import each other"), `registry.md` (plugins, default registries), the migration page, and the status table of `review/suggestions.md`: 26.2 can be closed (`sort_by_keys` falls back to `repr` and `slugify("😀")` returns a hash, both **[verified]** on `main`; `slugify` is deleted anyway). | no                                          |

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/utils/test_<module>.py`); the conventions of
`CLAUDE.md` apply (one behaviour per test, literal expected values, a regression test for every
bug, no local imports, `pytest.mark.parametrize` for the same check on several values).

- **Regression tests** (step 1): `int_repr(-(1 << 20000))` starts with `-0x`; `smallest_items` of an
  iterator of incomparable items returns the first `limit` items (and the function rejects a
  non-`Collection` through the type, not at run time); `LazySingleton` whose factory calls `reset`
  raises `RuntimeError` instead of hanging (the test uses the factory thread, no sleep).
- **`optional.py`** (step 4): one parametrized module over the 11 libraries and the pair. For each:
  `is_x_available` returns the value of `module_available`; `check_x` passes when it is available;
  `check_x` raises `MissingPackageError` whose message is the whole expected literal (including
  `pip install x` and `uv pip install x`) and whose `name` is the package; the cache is reset by
  `clear_availability_caches`. One test per behaviour, not a test per library.
- **`package.py`**: `check_package` and `raise_package_missing_error` give the same message with and
  without `install`; `package_available` does not import; `module_available` logs a broken install.
- **`LazySingleton`**: `set` replaces the value, `set` before `get` makes the factory unnecessary,
  `reset` after `set` rebuilds; the existing threads and fork tests stay in
  `tests/integration/utils/`.
- **Moved code** (step 8): the tests of the default registries and of the plugins move with the code
  to `tests/unit/registry` and `tests/integration/registry`, with the same assertions (the
  behaviour is unchanged).
- **Contract of the census** (`tests/integration/utils/test_usage.py`, D4): walks `src/coola`, collects
  the names imported from `coola.utils.*` outside `coola.utils`, and fails when a public name of a
  `coola.utils` module (its `__all__`) is neither imported by another package nor listed in a
  small allow-list of building blocks (`ForkAware`, `get_package_version`, `MissingPackageError`,
  `availability_cache`, `clear_availability_caches`). Expected cost: about 0.3 s for 225 files; it is a design gate, not a test of the exports
  of `__all__`.
- **`lint-imports`** (step 2): add the contract "`coola.utils` imports nothing from the other `coola`
  packages but `coola.validation`" (forbidden: every other package, including `registry` and `display`),
  and the contract "`coola.validation` does not import `coola.utils`".
- **Lazy imports** (`tests/integration/test_lazy_imports.py`, `tests/integration/utils/test_imports.py`):
  the existing tests must keep passing; add `import coola.utils.imports.optional` to the list of
  imports that must not import an optional library.
- **Doctests** (`xdoctest src/coola`) and `mkdocs build --strict` stay clean: every module of
  `coola.utils` has a line in `refs/utils.md`.
- **Coverage**: 100 % of `src/coola/utils` and of the moved modules (`--cov=coola`), no
  `# pragma: no cover` for a line that can be tested. The `fork.py` guard (`hasattr(os,
"register_at_fork")`) keeps its existing `no branch` pragma.
- **Size** (informative): `coola/utils` about 1,950 lines (6,048), tests about 4,000 lines (9,196).

## 9. Decisions for the maintainers

**D1. Delete the sixteen modules without a caller?** _Recommended: yes._ They are 1,711 lines of
source, 2,789 lines of tests, one section each of the guide, and a surface that `coola` does not need
(a subprocess, a password prompt, process-wide mutation). A utility package earns its place by its
callers (section 1, "Earned place"); "small and tested" is true of every function one could add.
Anyone who needs them finds them in git history (the last commit that has them is `dfb12b36`).
Alternatives: keep them in a separate distribution (a decision about another repository, not about
this one), or keep the three that look most reusable (`bloom_filter`, `lru`, `batching`) and delete
the rest. The first alternative costs a release process; the second keeps 468 lines and 583 test lines
that nothing calls, and blurs the rule.

**D2. Move the default registries and the plugin loader to `coola.registry`?** _Recommended: yes._
It is registry code (it copies registries, it loads entry points into them), and it is what
`LazySingleton` should not know. The cost is a changed import in seven packages, which `lint-imports`
checks. Alternative: split `singleton.py` in `singleton.py`, `default_registry.py` and `plugins.py`
inside `utils`; cheaper, but `utils` would keep a concept it should not know (F3).

**D3. One `optional.py` or one module per library?** _Recommended: one module._ The 11 modules differ
by a string; a single module makes the repetition visible and a parametrized test covers all of them.
The cost is 21 patches in tests and the `add_backend.md` recipe. Alternative: keep one module per
library with two functions each (about 35 lines): the same patch paths, the same recipe, 400 lines
instead of 330 and 11 test files instead of one.

**D4. Enforce "earned place" with a test?** _Recommended: yes, as an integration test._ A rule that
rests on discipline is the rule that produced F1: 16 modules were added to `utils` with their tests,
each for a reason that was true on the day. The test is 30 lines and fails when a public name has no
caller. Alternative: only the sentence in the guide and `CLAUDE.md`.

**D5. Change the hash of a huge negative integer?** _Recommended: yes._ `0x-1000…` is not a
representation of the number, and a hash is only stable inside a version of `coola`
(`hashing` states that the hashes may change between major versions). Only a negative integer with
more than 4,300 digits is affected. Alternative: keep the bug and document it; not recommended
because the fix is the only way to keep `int_repr` equal to `hex(value)` for the negative case.

**D6. Replace `import_object` by `pkgutil.resolve_name`?** _Recommended: yes._ Same format
(`"pkg.mod:name"`), same errors, and no second function named like the one of `coola.factory`. The
only difference is `"pkg.mod:"`, which returns the module instead of raising; nobody passes it.

**D7. Make `_reset_after_fork` public?** _Recommended: no._ It is a hook for the fork handler, not
part of the API of a registry or of `LazySingleton`. The protocol (`ForkAware`) says so. A callback
passed to `track_fork(obj, callback)` would need a `WeakMethod` for the same effect, with more code.

## 10. Not covered

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
  it leaves `utils` with the loader in step 8 and is then paid by `coola.registry`, not by
  `coola.utils`.

## Appendix: effect on the size of the package

| Area              | Lines today | Expected after  | Why                                                                                        |
| ----------------- | ----------- | --------------- | ------------------------------------------------------------------------------------------ |
| 16 unused modules | 1,711       | 0               | deleted (D1), `array` included                                                             |
| `tensor`          | 123         | 0               | deleted; about 40 lines move to `testing/marks.py` and `reducer/torch.py`                  |
| `conversion`      | 115         | 0               | moves to `nested` (about 100 lines)                                                        |
| `singleton`       | 428         | about 140       | `LazySingleton` stays; about 290 lines move to `registry/default.py` and `plugins.py`      |
| `format`          | 491         | about 200       | six functions instead of fourteen                                                          |
| `introspection`   | 322         | about 150       | five functions instead of nine                                                             |
| `path`            | 212         | about 100       | two functions instead of four                                                              |
| `mapping`         | 94          | about 40        | one function instead of three                                                              |
| `imports`         | 1,742       | about 600       | one mechanism and 330 lines for the libraries (section 3.3)                                |
| `fallback`        | 421         | about 340       | `factory.py` takes the package name; the modules lose their `raise_x_missing_error` import |
| other modules     | 389         | about 380       | `fork`, `integer`, `safe`, `selection`, `stats`, `version`, `__init__`                     |
| **`coola.utils`** | **6,048**   | **about 1,950** |                                                                                            |
| tests of `utils`  | 9,196       | about 4,000     | 2,789 lines of deleted modules, one parametrized module for `optional`, moved tests        |

The line counts are estimates from the code to delete and from the prototype; the gate is the
properties of section 1 and the checks of section 8, not the size.

## Appendix: the names that stay, and who calls them

| Name                                                                     | Callers                                                             |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| `track_fork`                                                             | `identifier`, `registry`, `LazySingleton`                           |
| `str_indent`, `repr_indent`, `str_mapping`, `repr_mapping`               | `display`, `registry`                                               |
| `str_mapping_line`, `repr_mapping_line`                                  | `display`                                                           |
| `int_repr`                                                               | `hashing`                                                           |
| `get_qualified_class_name`                                               | `hashing` (and `recursive` after step 6)                            |
| `get_type_constructor`                                                   | `recursive`                                                         |
| `is_same_type`                                                           | `io`, `random`                                                      |
| `supports_methods`, `has_generated_dataclass_eq`                         | `equality`                                                          |
| `sort_by_keys`                                                           | `display`, `format`                                                 |
| `add_uuid_suffix`, `sanitize_path`                                       | `io`                                                                |
| `safe_repr`, `safe_str`                                                  | `equality`, `summary`                                               |
| `smallest_items`                                                         | `summary`                                                           |
| `LazySingleton`                                                          | `equality`, `hashing`, `iterator`, `random`, `recursive`, `summary` |
| `quantile`                                                               | `reducer`                                                           |
| `compare_version`                                                        | `nested`                                                            |
| `is_<pkg>_available`, `check_<pkg>`                                      | 59 files                                                            |
| `package_available`                                                      | `equality`, `hashing`, `summary` (lazy fallback lookups)            |
| `lazy_exports`                                                           | eight packages (their `__init__`)                                   |
| `is_module_imported`                                                     | `factory`                                                           |
| `MissingPackageError`, `availability_cache`, `clear_availability_caches` | `imports` and its tests                                             |
