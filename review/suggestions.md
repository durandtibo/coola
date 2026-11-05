# Codebase review: suggestions (round 8)

This file **replaces the previous `review/suggestions.md`** (round 7, items 43 to 52). That version is
in git history (last touched by `bc4a251d`). Nothing below repeats an item of round 7 or of the
`review/<package>.md` design documents, except where an item is explicitly marked _carried over_.

The numbering **continues** the previous rounds (sections 53 to 66), so that an ID such as `54.1` in a
branch name or a commit message is unambiguous.

## How this review was done

Checkout of `main` at `abdd23d1` (`.venv`, Python 3.14.8; NumPy, pandas, polars, pyarrow, torch, xarray,
JAX, pydantic and SciPy at the versions of the lock file). Baseline, run on the clean checkout:

- `pytest tests/unit tests/integration`: 13855 passed, 34 skipped, 47 xfailed, **1 failed**
  (`test_real_mark_is_importable_in_a_generated_module`, the round-6 item 39.1, red again), in 37 s,
  with 100% line and branch coverage (6997 statements, 1672 branches).
- `pytest tests/properties`: 698 passed, 2 xfailed (strict markers).
- `ruff check src tests` and `ty check` are clean.
- `uv build` gives a correct sdist and wheel (`py.typed`, `LICENSE`, `README.md` present).

What was done:

1. **Status of round 7.** All the items 43.1 to 49 are merged (`edf465cf..a9ba8511`, PRs #2716 to #2734).
   I did not re-verify each fix one by one; the full suite is green except 39.1 and no probe below
   touched a fixed behaviour in a way that regressed. Still open from earlier rounds: 39.1 (item 53)
   and 34.1 (item 54.2).
2. **Probes**, about 450 one-line calls (scripts kept outside the repository): 75 stdlib and exotic
   values through `objects_are_equal` and `hash_object` (equality and hash must agree), 62 pandas, polars,
   pyarrow and xarray values (dtype, names, attrs, NA kinds, categoricals, chunking), `io`, `factory`,
   `identifier`, `random`, `nested`, `recursive`, `iterator`, `reducer`, `rounding`, `summary`.
3. **Mechanical checks** with AST scripts over `src` and `tests`: public API surface of the 19 packages,
   use of every `coola.utils` function by the rest of `src`, the rules of `CLAUDE.md` for the unit tests.
4. **Performance** of 35 operations at 1e5 to 1e6 items against the library baseline (`==`,
   `np.array_equal`, `DataFrame.equals`, `torch.equal`, `deepcopy`), plus a `cProfile` of `hash_object`.
5. **Thread safety**: 16 threads on the first use of the default registries, three times, in a fresh
   interpreter; 8 threads x 5000 snowflake ids. No error, no duplicate.
6. **Packaging, CI and docs**: the build, `pyproject.toml`, the version matrix of the CI, every
   backticked `coola.*` name of `docs/` and `README.md`.

The library is in very good shape: no crash on the 450 probes except the ones listed, the threads
behave, the security-sensitive paths (`factory`, `io`) are conservative and documented, the hashes are
independent of `PYTHONHASHSEED`, the CI covers the lowest and highest version of every optional
dependency but one (item 63). The findings are narrow: **two pandas cases where `objects_are_equal` is
`True` and `hash_object` differs** (54.1, 54.2), **a red test whose cause is the formatter** (53),
**`hash_object` that cannot hash what `objects_are_equal` compares** (55), and an **API surface that
has drifted** (58 to 61).

Priority: **P1** wrong result, crash on realistic input or a red test on `main`, **P2** inconsistent or
surprising behaviour, **P3** cleanup or design.

## Summary

| ID   | P   | Area         | Finding                                                                                                      |
| ---- | --- | ------------ | ------------------------------------------------------------------------------------------------------------ |
| 53   | P1  | tests        | _Carried over_ 39.1: red on `main`; docformatter rewrites the string, and it will do it again                |
| 54.1 | P2  | hashing      | `Series(["a", None])` equals `Series(["a", nan])` but the hashes differ (a `DataFrame` hashes the same)      |
| 54.2 | P2  | hashing      | _Carried over_ 34.1: categoricals with the same values and a different order of categories                   |
| 54.3 | P3  | equality     | `attrs` are compared for xarray, ignored for pandas and Arrow metadata: no stated rule                       |
| 55   | P2  | hashing      | `hash_object` rejects classes, functions, `re.Pattern`, `slice`, `Ellipsis`, `array.array`                   |
| 56   | P3  | iterator     | `dfs_iterate`, `bfs_iterate`, `dfs_items` depend on `PYTHONHASHSEED` for a set                               |
| 57   | P3  | registry     | `summarize` and `recursive_apply` keep up to 1024 dynamically created classes alive                          |
| 58   | P3  | api          | 5 packages export two names for the same registry getter; `Transformer` and `Node` mean two things          |
| 59   | P3  | api          | 12 public functions take options positionally; `load_json` and `save_json` disagree                          |
| 60   | P3  | utils        | 12 modules of `coola.utils` (1303 lines, about 2300 lines of tests) are used by nothing in the library       |
| 61   | P3  | jsonable     | `to_jsonable` cannot convert a data frame or a table; the tuple-key error names the wrong type               |
| 62   | P3  | small gaps   | `factory` prefix with a trailing dot; `summarize(max_characters=0)`; rounding modes; `LazyFrame` message     |
| 63   | P3  | ci           | SciPy is missing from `dev/config/package_versions.json`: its lower bound is never tested                    |
| 64   | P3  | tests        | 117 unit tests with several assertions, 18 with a loop; the property tests have no object column (54.1)      |
| 65   | P3  | performance  | Measured costs, and the three places worth a profile                                                         |
| 66   | -   | follow-up    | Proposed order of the work                                                                                   |

## 53. The red test is a formatter problem, and it will come back (P1, _carried over_ 39.1) **[verified]**

`tests/integration/testing/test_marks.py::test_real_mark_is_importable_in_a_generated_module` fails on
`main`. The generated module is not valid Python:

```python
pytester.makepyfile(
    """From coola.testing.marks import numpy_available,
    numpy_not_available.

    @numpy_available
    def test_with_numpy(): ...
    ...
    """
)
```

`From` is capitalised and the import line is wrapped and ends with a full stop. The cause is the
`docformatter` hook of `.pre-commit-config.yaml` (and `bot-format-files.yaml`): it takes the string
literal that is the only argument of the call for a docstring and rewrites it as prose. Reproduced with
`docformatter --config ./pyproject.toml --diff` on a copy of the correct source. The history confirms
it: `f9ade729` (#2696) fixed the source, `bc9e1534` (#2713, "Minor improvements") ran the formatter and
broke it again. Fixing the text a third time is not a fix. What docformatter leaves alone is a
module-level constant (checked: `SOURCE = """..."""` is unchanged, a keyword argument is not):

```python
SOURCE = """
from coola.testing.marks import numpy_available, numpy_not_available

@numpy_available
def test_with_numpy(): ...
"""


def test_real_mark_is_importable_in_a_generated_module(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(SOURCE)
    pytester.runpytest().assert_outcomes(passed=1, skipped=1)
```

This is the only one of the seven `makepyfile` calls of `tests/` whose source starts right after the
opening parenthesis. A CI that is red on `main` hides real failures, so this is the first PR.

## 54. Equality and hashing disagree on three pandas cases

The contract is "equal objects have the same hash". The round-7 differential fuzz found no violation
except 34.1 because it did not generate object columns that mix `None` and `nan`.

### 54.1 `None` and `nan` in an object column **[verified]** (P2)

```python
import numpy as np, pandas as pd
from coola.equality import objects_are_equal
from coola.hashing import hash_object

a = pd.Series(["a", None], dtype=object)
b = pd.Series(["a", np.nan], dtype=object)
objects_are_equal(a, b)  # True   (without equal_nan; pandas: a.equals(b) is True too)
hash_object(a) == hash_object(b)  # False

da, db = a.to_frame("x"), b.to_frame("x")
hash_object(da) == hash_object(db)  # True: the frame and the series do not agree
```

The Series hasher and the DataFrame hasher normalise the missing values differently, so the same
column hashes differently depending on its container. `generate_stable_content_id` on a series is then
different for two series that the library calls equal. Proposal: one function in `hashing/pandas.py`
that maps `None`, `nan`, `NaT` and `pd.NA` of an object array to one tag, used by both hashers, and a
test that hashes `Series(x)` and `DataFrame({"c": x})` of the same data. Whether `objects_are_equal`
should call `None` and `nan` equal when `equal_nan=False` is a separate question; the pandas answer is
yes, and the test of the tester should say so.

### 54.2 Categories in a different order **[verified]** (P2, _carried over_ 34.1)

```python
a = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"]))
b = pd.Series(pd.Categorical(["a", "b"], categories=["b", "a"]))
objects_are_equal(a, b)  # True  (a.equals(b) is True)
hash_object(a) == hash_object(b)  # False
```

Still open since round 6. The strict xfail `test_different_frames_have_different_hashes` ("pandas
metadata") in `tests/properties/test_library_types.py` is exactly this case: I enumerated its 72 specs
pair by pair, and the only 4 pairs where equality and hash disagree are the pairs that differ by the
order of the categories of an unordered categorical (`["a", "b"]` against `["b", "a"]`). So the test
is right and its reason is misleading. Pick the side where the change is smallest: for an **unordered**
categorical, hash the decoded values and the sorted categories; for an **ordered** one, keep the order
(the order is part of the meaning). Then remove the xfail, and rename its reason to "34.1".

### 54.3 Metadata is compared or ignored depending on the library (P3) **[verified]**

| Library | `attrs` / metadata                                  | Compared by `objects_are_equal` |
| ------- | --------------------------------------------------- | ------------------------------- |
| xarray  | `DataArray.attrs`                                   | yes                             |
| xarray  | `DataArray.encoding`                                | no                              |
| pandas  | `DataFrame.attrs`                                   | **no**                          |
| Arrow   | `Table` schema metadata                             | **no**                          |
| polars  | (no metadata)                                       | -                               |

Each choice is defensible; the missing piece is the rule. Add one sentence per library to the user
guide ("equality compares what the library's own `equals` compares, plus names and dtypes") and a
test per row so that a library upgrade that changes `equals` is noticed.

## 55. `hash_object` cannot hash what `objects_are_equal` compares (P2) **[verified]**

`objects_are_equal(x, x)` is `True` and `hash_object(x)` raises `TypeNotRegisteredError` for: a class
(`int`), a function or a lambda, a module, `re.compile("a")`, `slice(1, 2)`, `...`, `NotImplemented`,
`array.array("i", [1])`, an object without `__eq__` or dataclass, a `functools.partial`, a lock, an
iterator. `memoryview`, `UUID`, `timedelta`, `deque`, `ChainMap`, `mappingproxy` work.

The ML use case is a config dict that holds a class or a function (`{"model": nn.Linear, "loss":
F.mse_loss, "pattern": re.compile(...)}`): today the caller must pass `ignore_unhashable=True`, and then
two configs with a different `model` get the same identifier. That is a worse outcome than the error.
The stable ones can be hashed without surprises:

- a class or a named function: `module:qualname` (a lambda or a local function has no stable name and
  keeps raising);
- `re.Pattern`: `(pattern, flags)`; `slice`: `(start, stop, step)`; `Ellipsis`, `NotImplemented`: a tag;
- `array.array`: `(typecode, bytes)`.

Add them to the default registry as small `ContentHasher`-style classes, and document in
`uguide/hashing.md` the list of what is rejected on purpose (locks, iterators, plain objects).
`ignore_unhashable=True` should skip a value (today it also hides the three cases above).

## 56. Traversal order of a set depends on `PYTHONHASHSEED` (P3) **[verified]**

```text
PYTHONHASHSEED=1  list(dfs_iterate({"a", "b", "c", "d"}))  ->  ['a', 'd', 'b', 'c']
PYTHONHASHSEED=2                                           ->  ['c', 'b', 'a', 'd']
PYTHONHASHSEED=3                                           ->  ['c', 'b', 'd', 'a']
```

`dfs_items` gives the same instability in the paths (`('k', 'c')` first or last). The user guide says
"the order inside a set is not specified", so this is documented, but it is now the only package whose
output changes between runs: `hashing`, `summary` (43.1) and `jsonable` order a set. `smallest_items`
and `sort_by_keys` already exist; use a `sorted_or_insertion_order(set)` for the child finder of
`set` and `frozenset`. Cost: a sort of the set, only for sets. It also makes `round_values` and
`recursive_apply` deterministic for a set of unorderable items (they rebuild a set, so only the
iteration order of the callback changes).

## 57. Registries keep dynamically created classes alive (P3) **[verified]**

`TypeRegistry` caches the resolution per type in a dict that is reset at 1024 entries
(`registry/type.py:731`). The key is a strong reference, so up to 1024 classes (with their methods, their
module globals and everything they close over) survive after the last instance is gone:

```python
for i in range(3000):
    C = type("C", (), {})
    summarize(C())
# 952 of the 3000 classes are still alive after gc.collect()
```

`recursive_apply` shows the same (50/50 alive); `objects_are_equal`, `hash_object` and `to_jsonable` did
not on the same loop (identity shortcut or no cache hit). Services that build a class per request
(`pydantic.create_model`, `make_dataclass`, `namedtuple`) hold on to them. Use a
`weakref.WeakKeyDictionary` for `_cache` (types are weak-referenceable), or cache only the types whose
`__module__` is not `__main__` or a builtin. Test with `weakref.ref(C)` after `gc.collect()`; no sleep
is needed.

## 58. The names of the public API have drifted (P3) **[verified]**

1. **Two names for one getter.** `hashing`, `summary`, `recursive`, `iterator` and `random` export both
   `get_default_registry` and `get_default_<x>_registry` (`get_default_hasher_registry`,
   `get_default_summarizer_registry`, `get_default_transformer_registry`,
   `get_default_child_finder_registry`, `get_default_random_manager_registry`); `equality`, `jsonable`
   and `rounding` export one. Pick the short name (the packages are already namespaced) and delete the
   other: breaking changes are accepted, and 5 `__all__` and as many docs lines get shorter. The same
   goes for `hashing.get_default_registry` vs `get_default_hasher_registry` in the docs.
2. **Same name, two meanings.** `coola.recursive` and `coola.jsonable` both define `Transformer`,
   `TransformerRegistry`, `DataclassTransformer`, `MappingTransformer`, `SequenceTransformer`,
   `SetTransformer` and `register_transformers`; they are different protocols (`transform(data,
   func)` vs a converter to JSON). `from coola.recursive import TransformerRegistry` and `from
   coola.jsonable import TransformerRegistry` give two unrelated classes, and an IDE import suggestion
   picks one at random. Rename the `jsonable` ones to `Converter`, `ConverterRegistry`,
   `register_converters`. Same for `Node`, exported by both `hashing` and `summary` with different fields
   (`hashing.Node`, `summary.Node`): rename to `HashNode` and `SummaryNode`, or stop exporting them
   (they are an engine detail).
3. **`hashing` has three similar names** (`StrHasher`, `StringHasher`, `ReprHasher`, plus
   `BytesHasher`, `ContentHasher`): `StringHasher` is for `str` and `UserString`, `StrHasher` hashes
   `str(obj)` and `ReprHasher` `repr(obj)`. The docstrings say it, but a name like `StrHasher` next to
   `StringHasher` is easy to confuse; a table "type -> hasher -> what is hashed" in `uguide/hashing.md`
   or a rename of `StrHasher` to `StrOfHasher` would help.
4. **`to_jsonable(..., on_unsupported: str)`** is typed `str`; every sibling uses `Literal[...]`
   (`round_values`, `flatten_mapping`, `merge_mappings`, `get_secret_hashers`).
5. **`__all__` of `utils`, `testing`, `formatting` is empty** by design (import from the module). It is
   the only place where `from coola.formatting import truncate_str` does not work while `from
   coola.nested import flatten_mapping` does. Either re-export the formatting functions (they are
   documented as user-facing in `uguide/formatting.md`) or say it in the first line of the page.

## 59. Options that are still positional (P3) **[verified]**

`CLAUDE.md`: "Keep new options keyword-only (`*`), as `hash_object` does." A scan of the public
functions with a default value that is not keyword-only gives:

| Function                                       | Positional option(s)                         |
| ---------------------------------------------- | -------------------------------------------- |
| `io.load_json`, `io.load_text`                 | `encoding`                                   |
| `nested.add_prefix_suffix_to_keys`             | `prefix`, `suffix` (both default to `""`)    |
| `nested.flatten_frame`                         | `separator`, `depth`, `expand_lists`         |
| `hashing.tagged`                               | `payload`                                    |
| `hashing.get_secret_hashers`                   | `on_secret`                                  |
| `formatting.mask_secret`                       | `show_first`, `show_last`                    |
| `formatting.truncate_str`                      | `max_len`, `suffix`                          |
| `formatting.unified_diff`                      | `label_a`, `label_b`, `context_lines`        |
| `formatting.make_bar`, `str_human_byte_size`   | `length`, `unit`                             |
| `factory.resolve_object`                       | `cls` (default `object`)                     |
| `summary.format_fields`                        | `max_fields` (no default, but unnamed at the call sites) |

The asymmetry that bites a user is `save_json(obj, path, *, encoding=..., allow_nan=...)` against
`load_json(path, encoding=None)`: the two functions that are used together have a different calling
convention, and `load_json(path, "utf-8")` works while `save_json(obj, path, "utf-8")` is a
`TypeError`. Make them all keyword-only in the same release as 58.1, with the migration note.
(`add_prefix_suffix_to_keys("x_")` is the only one where the positional form is the documented
example; keep `prefix` positional-or-keyword and make `suffix` keyword-only.)

## 60. A tenth of `coola.utils` is not used by the library (P3) **[verified]**

An AST scan of every public function of `src/coola/utils/` for a use outside its own module gives no
use at all for: `bloom_filter` (224 lines), `logging` (50), `git` (153), `batching` (51), `file_size`
(53), `env_vars` (204), `timing` (92), `lru` (204), `password` (69), `recursion` (46), `array` (58),
`date` (99): **1303 lines of source, about 2300 lines of unit tests, a 449-line user guide page and a
page of the developer guide**, for helpers that have nothing to do with comparing, hashing or
summarizing. They are exported as public API (`utils/__init__.py` lists them as "standalone helpers"),
so each is a promise: `get_password` and `run_git_command` in particular are the kind of function
whose behaviour a user will rely on. The cost of keeping them is the CI time, the 100%-coverage rule,
the docs and the review time of every change to the registry or imports they never use.

Options, from the smallest: (1) keep and declare them **private** (`coola.utils._standalone`) so that
no promise is made, (2) move them to a separate package (`coola-utils`) and depend on it only in the
dev group, (3) delete. With a major release open, (3) for `git`, `password`, `date`, `env_vars`,
`lru`, `bloom_filter`, `timing`, `logging`, `batching`, `file_size` and (1) for `array`/`recursion` if
a test fixture needs them. Check with `grep -rn "coola.utils"` in the dependent projects of the author
before the deletion.

## 61. `to_jsonable` and the data frames (P3) **[verified]**

13 cells of `tests/integration/test_support_matrix.py::test_support_matrix_to_jsonable` are `xfail`
("known gap"): `pd.DataFrame`, `pl.DataFrame`, `pa.Table`, `pd.Categorical`, `pd.Period`,
`pd.Interval`, `re.Pattern`, `slice`, `timezone`, `torch.dtype`, `torch.device`, `torch.sparse`,
`torch.meta`. The data frames are the ones a user actually puts in a result dict before `save_json`.
The pattern of the existing converters (`ndarray` -> nested list) suggests
`DataFrame -> {"columns": [...], "data": [[...]]}` (or `dict` of columns, one choice documented), a
`Categorical`/`Period`/`Interval` -> its string or list, a dtype or device -> `str(x)`, a `Pattern` ->
`.pattern`. At minimum, `on_unsupported="repr"` as a third value is a one-line escape that makes
`save_json` of a result dict possible for the 13 types.

Also, in a dict with a tuple key the error is wrong about the type the user passed:

```python
to_jsonable({(1, 2): 3})
# TypeError: A JSON key must be a str, bool, int, float or None after the conversion,
#            but received a key of type list
```

The key was a `tuple`, converted to a `list` before the check. Report both the original and the
converted type, and say that the key was converted.

## 62. Small gaps (P3)

1. **A `factory` prefix with a trailing dot never matches** **[verified]**.
   `import_object("coola.io.JsonLoader", allowed_prefixes=["coola."])` raises "not allowed" while
   `["coola"]` and `["coola.io"]` work (the match is by dotted component, `factory/instantiation.py:89`).
   `"coola."` is the natural way to write "everything under coola". Validate the prefixes once
   (`validate_not_empty`, no leading or trailing dot) or strip one trailing dot, and say so in the
   docstring. Related and documented: `allowed_prefixes=None` allows every importable callable
   (`factory("shutil.which", "ls")` runs). That is the right default for a library API; consider a
   `COOLA_FACTORY_ALLOWED_PREFIXES`-style process-wide default for applications that read configs from
   files, so the allow-list does not have to be threaded through every call.
2. **`summarize(..., max_characters=0 or 1)` truncates the keys** **[verified]**.
   `summarize([1, 2, 3], max_characters=0)` gives `(): ` for each item (the index disappears), and
   `summarize({"abc": "defghijkl"}, max_characters=1)` gives `(.): .`. The limit is documented for
   the value; apply it to the value only, or reject a limit under the header length with the
   `validate_*` helpers.
3. **`round_values` modes** **[verified]**. `mode="trunc"` raises (`must be one of ('round', 'floor',
   'ceil')`), `decimals=-2` still raises (item 45.4 listed it; #2729 did not change it), and
   `round_values(2.5)` is `2.0` and `round_values(0.5)` is `0.0` (half to even, as `round` and NumPy). The
   docs should say "half to even" next to `mode="round"`, and `"trunc"` is a three-line rounder for each
   backend. Negative `decimals` is the missing one for large numbers (`round_values(1234.5, decimals=-2)`).
4. **`LazyFrame` message** **[verified]**. `objects_are_equal(pl.LazyFrame(...), pl.LazyFrame(...))`
   raises "`==` raised or returned a value that cannot be converted to a bool ... Register an equality
   tester for this type", while `summarize` still has a `PolarsLazyFrameSummarizer`. A dedicated message
   ("call `.collect()` first; a LazyFrame is never collected by `coola`") costs one tester that raises,
   and matches the summary's rule "a lazy frame is never collected".
5. **`deque(maxlen=...)` is ignored** by equality and hashing (`deque([1], maxlen=2)` equals and hashes
   like `deque([1])`). Same behaviour as `==`, so probably right; one sentence in the guide.
6. **`recursive_apply` loses sharing** **[verified]**: `s = [1]; recursive_apply([s, s], f)` returns two
   distinct lists (`r[0] is r[1]` is `False`), as `deepcopy` does not. Document it next to the cycle rule
   (the cycle raises, the DAG is expanded).

## 63. SciPy is not in the version matrix of the CI (P3) **[verified]**

`dev/config/package_versions.json` lists 12 packages (colorlog, jax, numpy, packaging, pandas, polars,
pyarrow, pydantic, pytest, rich, torch, xarray) and **not `scipy`**, although `pyproject.toml` has the
extra `scipy = ["scipy>=1.13"]` and `equality/tester/scipy.py` is 270 lines. `dev/generate_versions.py`
reads the dependencies from `pyproject.toml`, so the file is stale (the bot probably ran before the extra
was added) or the bot did not run. Run `dev/generate_versions.py`, commit the file, and add a CI check
that every key of `[project.optional-dependencies]` has an entry in the JSON, so that the next extra
cannot be forgotten. Today the sparse-matrix tester has never been run against `scipy==1.13`.

## 64. The tests (P3)

1. **`CLAUDE.md` rules that the unit tests still break** (AST scan of `tests/unit`): **117 tests have
   more than one `assert` or `pytest.raises`** (top directories: `factory` 13, `utils` 12, `imports` 12,
   `hashing` 11, `registry` 11, `rounding` 8; for example `test_text_saver_save_rejects_non_str_*`,
   `test_base_file_saver_save_failure_leaves_the_directory_empty`) and **18 contain a loop** (`nested`
   6, `hashing` 4, `equality` 3). No test has a local import, a `pytest.raises` without `match`, or a
   `time.sleep` (it is patched); that part of the guide is respected. A first pass: split the 117 (a
   mechanical edit, one new test name each), and move the loops to `parametrize`.
2. **The property tests do not generate the data of 54.1.** The pair test of 54.2 exists (as a strict
   xfail); `make_frame` has no object column. Add an object-column strategy mixing `None`, `nan`,
   `NaT` and `pd.NA`, and a test over pairs with `assume(objects_are_equal(a, b))` then
   `hash_object(a) == hash_object(b)`, for both `Series` and `DataFrame`. 54.1 falls out of it in a few
   hundred examples. This is how the differential fuzz of round 7 should be kept: as a regression test
   and not as a throw-away script.
3. **47 xfails are a list of known gaps** (13 `to_jsonable`, torch string comparison, two property
   tests). With strict markers (#2733) they cannot pass silently, which is the right design. Each
   `reason="known gap"` should name an issue or a section of this file, so that a gap that is decided
   "will not fix" turns into a documented limitation and leaves the xfail list.
4. **The size of the suite**: 75k lines of tests for 33k lines of `src` (equality: 17k for 6.4k). The
   run time (37 s) is fine; the maintenance cost is in the shape: the same registry/`find`/`has`/
   `register` tests are repeated in each of the 8 registry packages. A shared `registry_contract`
   fixture (a parametrised class in `tests/helpers.py`) would test the `TypeRegistry` behaviour once
   per subclass, in 20 lines instead of about 250.

## 65. Performance: measured costs (P3)

Best of 3, Apple M-series, Python 3.14. `ref` is the plain library call.

| Operation                                          | coola    | ref        | ratio |
| -------------------------------------------------- | -------- | ---------- | ----- |
| `objects_are_equal`, list of 2e5 ints              | 7.7 ms   | 0.1 ms     | 77x   |
| `objects_are_equal`, dict of 2e5 str -> int        | 33.6 ms  | 3.5 ms     | 10x   |
| `objects_are_equal`, 5e4 small nested dicts        | 138 ms   | 2.2 ms     | 63x   |
| `objects_are_equal`, ndarray 1e6 / DataFrame 1e6   | 0.4 / 1.3 ms | 0.4 / 0.4 ms | 1x / 3x |
| `objects_are_equal`, torch 1e6 / polars 1e6        | 0.1 / 2.7 ms | 0.1 / 0.3 ms | 1x / 9x |
| `hash_object`, 5e4 small nested dicts              | 423 ms   | -          | 8.4 us per dict |
| `hash_object`, list of 2e5 floats / str            | 139 / 84 ms | -       | 0.7 / 0.4 us per item |
| `hash_object`, ndarray / DataFrame / tensor 1e6    | 8.8 / 9.5 / 4.4 ms | - | memory bound |
| `round_values(recursive=True)`, 5e4 small dicts    | 359 ms   | `recursive_apply` 103 ms | 3.5x |
| `to_flat_dict`, 5e4 small dicts                    | 177 ms   | -          |       |
| `summarize`, 5e4 small dicts                       | < 1 ms   | -          | bounded, as designed |

Nothing is a problem for the documented use (test assertions, identifiers of configs), and the array
and table paths are at the speed of the library call. Three places are worth a profile if the goal is
speed: (1) the engine overhead per node of `objects_are_equal` (2.7 us per node, and a
`cProfile` of `hash_object` shows 80000 calls of `fill` and 140000 of `tagged` for 20000 dicts: the
cost is spread over `fill`, `leave`, `tagged`, `int_repr` and `isinstance`, so a gain needs a fast path
for `str` and `int` leaves like the one added for floats in #2735, not a micro-optimisation);
(2) `round_values(recursive=True)` costs 3.5x its own traversal, so the per-leaf dispatch (a registry
lookup on each leaf, 200000 here) is the part to cache by type for the duration of the call; (3) the
`hash_object` of a list of floats at 0.7 us per item, which could use `array.array("d", ...).tobytes()`
for a homogeneous list, as #2735 did for equality.

## 66. Proposed order of the work

One branch and one pull request each, as `CLAUDE.md` asks:

1. `test/fix-generated-module-source` - item 53 (turns the CI green; do this first).
2. `bugfix/hash-pandas-na-and-categories` - items 54.1 and 54.2, with the pair-based property tests (the
   xfail of 54.2 becomes a plain test, 64.2 adds 54.1) as the regression tests (check that they fail on `main`).
3. `ci/add-scipy-to-versions` - item 63.
4. `feat/hash-classes-functions-patterns` - item 55.
5. `refactor/api-names` - items 58, 59 together (one breaking change and one migration note); then
   `refactor/remove-unused-utils` - item 60.
6. `bugfix/deterministic-set-traversal` (56), `perf/weak-registry-cache` (57), then the small gaps
   of 61 and 62, one PR per numbered point.
7. `test/split-multi-assert-tests` - 64.1, in one PR per directory.

Not changed by this review: nothing in `src`, `tests` or `docs`. Only this file.
