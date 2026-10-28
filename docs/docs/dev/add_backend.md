# How to Add Support for a New Library

:book: This page is the checklist to add support for a new optional library (a "backend", such as
NumPy, PyTorch or pandas) to `coola`. A backend is supported by several packages, each with its own
registry, and it is easy to add the equality support and forget the others.

To teach `coola` about **one of your own classes**, without changing `coola`, see
[How to Add Support for a New Type](../uguide/extending.md) instead.

In this page, `<pkg>` is the name of the library, e.g. `pandas`.

## 1. Declare the optional dependency

1. Add the extra to `[project.optional-dependencies]` of `pyproject.toml`, e.g.
   `<pkg> = ["<pkg>>=1.0"]`. The `all` extra, the CI jobs that test each dependency with its
   supported versions, and `dev/config/package_versions.json` (updated by a bot) follow from it.
2. Add `is_<pkg>_available` and `check_<pkg>` to `src/coola/utils/imports/optional.py`, with the same two
   lines as the other libraries (`module_available("<pkg>")`, and `raise_package_missing_error`
   when it is not available). Decorate `is_<pkg>_available` with `availability_cache` (not with
   `functools.cache`), so that `clear_availability_caches` knows about it. Export the two names in
   `src/coola/utils/imports/__init__.py`.
3. Create `src/coola/utils/fallback/<pkg>.py`, copied from `pandas.py`: a fake module with fake
   classes that raise `MissingPackageError` (`make_fake_class("<pkg>")`), used when the library is not installed. It must have
   every attribute of the library that is used in an annotation or a `class X(Base[pkg.Type])`
   statement, because these are evaluated when the module is imported.
4. Add the marks for the tests: add one row to `_DEPENDENCIES` in `src/coola/testing/marks.py`
   (the `is_<pkg>_available` function and the name used in the skip reasons). `__all__` is computed
   from the table and the marks are typed by the module `__getattr__`, so nothing else changes:

```python
"<pkg>": _Dependency(is_<pkg>_available, "<pkg>"),
```

Do not call `requires(is_<pkg>_available(), "<pkg>")` at module level: it runs the check, and
so imports the library, as soon as `coola.testing.marks` is imported. The table stores the
function, and the marks are built on first use.

## 2. Equality

1. Create `src/coola/equality/tester/<pkg>.py`. For a library of tables (data frames, series,
   columnar arrays), write an adapter (follow `pandas.py`, `polars.py` or `pyarrow.py`): a class
   with a `kind_name` (the plural word used in the reasons), a `same_structure(other, ctx)` method
   and a `columns()` method that yields `Column`s, and a function that builds it from an object.
   A column is `numeric` or `duration` (compared by the numeric kernel, with the tolerance and
   `equal_nan`; the null values are given as a mask) or `exact` (compared by the `equals` of the
   library). Register `TableEqualityTester(adapter)` for the types of the library.
2. For an array library, write a tester that derives from `BaseEqualityTester` (follow
   `numpy.py`): it checks the data type and the shape, then compares the values with the shared
   kernel, and returns a verdict (`ctx.fail(reason, *path)` returns `False`) or the child pairs.
   For an optional library, `__init__` calls `check_<pkg>()`. Export the tester in
   `equality/tester/__init__.py` as a lazy name.
3. Register the tester: add a `get_default_testers` function to `equality/tester/<pkg>.py` that
   returns the `{type: tester}` mapping of the library (and returns `{}` if `is_<pkg>_available()`
   is `False`), and add the library to `_LAZY_TESTERS` in `equality/tester/interface.py`.
4. Document the behavior of the library in the table
   ["Behavior by type"](../uguide/equality.md#behavior-by-type-nan-and-tolerance).

## 3. The other packages

Decide for each package whether the library needs an entry, and write the decision down in the
pull request.

| Package           | What to do                                                                                                                                                                                                                                                                         | Default                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `coola.summary`   | Add a `BaseMetadataSummarizer` in `summary/<pkg>.py` that describes the metadata (shape, dtype, columns) in `_describe`, export it as a lazy name, and register it with a `get_default_summarizers` function and the `_LAZY_SUMMARIZERS` table of `summary/interface.py`.          | Without it, the summary is the `str` of the object, which can be very long. |
| `coola.hashing`   | Nothing by default: there is no canonical byte form for arrays and data frames. Document how to register a hasher with `register_hashers`.                                                                                                                                         | The type is not hashable.                                                   |
| `coola.recursive` | Nothing: the objects of the library are leaves, passed to the function.                                                                                                                                                                                                            | Leaf.                                                                       |
| `coola.iterator`  | Nothing: the objects are leaves, and are not expanded.                                                                                                                                                                                                                             | Leaf.                                                                       |
| `coola.random`    | If the library has a global random number generator, add a `BaseRandomManager` in `random/<pkg>.py`, and register it in `random/interface.py` (the manager is imported with `import_object`, because the RNG of the library must be seeded even if the library is imported later). | Not managed.                                                                |
| `coola.reducer`   | If the library can compute statistics, subclass `BaseReducer` in `reducer/<pkg>.py` (the six computations, `_sort`, optionally `_prepare`). Its results must agree with `NativeReducer` (see `tests/unit/reducer/test_consistency.py`).                                            | Not supported.                                                              |

## Lazy imports

Importing `coola` must not import the optional libraries, because it can take seconds (for example
`torch`). The modules that need a library import it at the top, as usual, with the fallback module
when it is not installed. What is lazy is _who imports these modules_:

- **Package exports.** The `__init__.py` of a package lists the names that are defined in a module
  of an optional library in a `_LAZY_EXPORTS` table, and defines `__getattr__` and `__dir__` with
  `lazy_exports`. The names stay in `__all__`, and `from coola.equality.tester import
TorchTensorEqualityTester` works as before, but the module is imported when the name is used.
  Import the names for the type checkers in the `if TYPE_CHECKING:` block.
- **Registries.** The default registries do not import the modules of the optional libraries.
  `registry.register_lazy(packages, "package.module:get_default_...")` registers a group of entries
  that is built the first time the registry looks for a type whose class (or one of its base
  classes) comes from one of the `packages`, or needs the whole registry (`len`, iteration, `items`).
  An object cannot have a type of a library that was not imported, so a library is only imported by
  the registry if the program already imported it. The entries that are already registered are not
  replaced.
- **Code that handles the objects of a library only if they exist.** Use
  `sys.modules.get("numpy")` instead of importing the library (see `coola.utils.conversion`).

## 4. Tests and documentation

1. Write the unit tests next to the existing ones, with the mark of step 1
   (`@<pkg>_available`), and an import guard for the library as in the other test files. Test with
   and without the library: the CI runs the tests with the required dependencies only.
2. Every new public class and function must be in the `__all__` of its module and of its package:
   `tests/unit/test_public_api.py` checks it.
3. Check that importing `coola` and its subpackages does not import the library: add the library
   to `OPTIONAL_LIBRARIES` of `tests/integration/test_lazy_imports.py`.
4. Update the user guide pages of the packages you touched, and the "Supported Types" list of
   `docs/docs/uguide/equality.md`.

## Summary of the files

| Step              | Files                                                                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dependency        | `pyproject.toml`, `utils/imports/optional.py`, `utils/imports/__init__.py`, `utils/fallback/<pkg>.py`, `testing/marks.py`                                          |
| Equality          | `equality/_helpers/<pkg>.py` (the kernel, if the library has arrays), `equality/tester/<pkg>.py`, `equality/tester/__init__.py`, `equality/tester/interface.py` |
| Summary           | `summary/<pkg>.py`, `summary/__init__.py`, `summary/interface.py`                                                                                               |
| Others (optional) | `random/<pkg>.py`, `random/interface.py`, `reducer/<pkg>.py`                                                                                                    |
| Tests and docs    | `tests/unit/...`, `docs/docs/uguide/*.md`                                                                                                                       |
