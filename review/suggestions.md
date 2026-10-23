# Codebase review: suggestions (round 4)

This file **replaces the previous `review/suggestions.md`** (round 3, 543 lines, items 1.1 to 11).
That version is in git history (last touched by `55086b7f`). Section 20 gives the status of its
items. Nothing below repeats an item of round 3, `src-coola-review.md`, `noqa-review.md` or
`python-311-modernization.md`, except where an item is explicitly marked _carried over_.

The numbering **continues** the previous rounds (sections 12 to 20), so that an ID such as `13.2` in a
branch name or a commit message is unambiguous.

## How this review was done

Checkout of `main` at `0287a61a` (`.venv`, Python 3.14.8, NumPy, torch, pandas, polars, pyarrow,
xarray, JAX, pydantic, SciPy installed). Baseline: `pytest tests/unit tests/integration` gives
12237 passed, 40 skipped, 3 xfailed in 26 s with 99% line coverage (6149 statements, 0 missed, 1
partial branch); `ruff check src tests` and `ty check` are clean. The review changed no source file.

1. **Read-through.** Every module of `src/coola` (207 files, about 30.5k lines) was read with the
   docstrings stripped, package by package: `registry`, `equality`, `hashing`, `identifier`, `utils`,
   `io`, `summary`, `iterator`, `recursive`, `nested`, `reducer`, `random`, `display`, `factory`,
   `validation`, `testing`. The tests (311 files, 62.9k lines), the docs and the CI were surveyed with
   scripts (conventions, structure, coverage), not read line by line.
2. **Probes.** Items marked **[verified]** were reproduced on this checkout with a script and the
   repro is given. Items marked **[reading]** or **[lead]** come from reading the code and were not
   reproduced (GPU, Windows and CI-only behaviour were not run).
3. **Support matrix.** About 100 representative types (NumPy, torch, pandas, polars, pyarrow, xarray,
   JAX, SciPy, pydantic, stdlib containers and scalars) were run through `objects_are_equal`
   (against a deep copy), `hash_object`, `summarize`, `dfs_iterate` and `recursive_apply`. The gaps
   are in 14.4 and 18.1.

Almost all round-3 items were merged in the meantime (PRs #2070 to #2235, section 20). The new
findings are mostly in two places: the lazy registries of #2231 (section 12), and the places where
`equality`, `hashing` and `summary` treat the same type differently (sections 13 to 15).

Priority: **P1** wrong result or crash on realistic input, **P2** inconsistent or surprising
behaviour, **P3** cleanup or design.

## Summary

| ID   | P   | Area     | Finding                                                                                         |
| ---- | --- | -------- | ----------------------------------------------------------------------------------------------- |
| 12.1 | P1  | registry | `in`, `has`, `get`, `[]`, `find` with a non-type raise `AttributeError` (regression of #2231)   |
| 14.1 | P1  | hashing  | NumPy structured dtypes with different field names or types hash the same                       |
| 14.2 | P1  | hashing  | pandas objects that differ by index name, `columns.name` or categories hash the same            |
| 15.1 | P1  | summary  | `summarize` of a collection at the depth limit raises when an item cannot be converted to `str` |
| 12.2 | P2  | registry | `repr`, `len` or iteration of a default registry imports every installed optional library       |
| 12.4 | P2  | hashing  | `HasherRegistry` fallbacks: slow (16 µs per dataclass), ignored by `equal`, not locked          |
| 13.1 | P2  | equality | No pydantic tester, and an exception raised by `==` itself escapes `ObjectEqualHandler`         |
| 13.2 | P2  | equality | `objects_are_allclose` is asymmetric for NumPy and torch, symmetric for scalars, pandas, polars |
| 13.4 | P2  | equality | `compare` and `assert_objects_equal` raise when a compared field has a failing `__str__`        |
| 14.3 | P2  | hashing  | Arrays and tensors with different NaN bit patterns hash differently, floats do not              |
| 14.4 | P2  | hashing  | Types supported by equality are not hashable; wrong exception for `np.timedelta64` and `pd.NaT` |
| 14.6 | P2  | hashing  | The "stable" IDs have no golden vectors and no format version                                   |
| 14.7 | P2  | hashing  | Equal dataclasses hash differently when a field has `compare=False`                             |
| 16.1 | P2  | io       | `save_*` and `load_*` raise `AttributeError` for a `str` path                                   |
| 16.3 | P2  | factory  | `resolve_saver`/`resolve_loader` call the target before checking its type                       |

## 12. `registry` and lazy loading

### 12.1 Non-type keys raise `AttributeError` once a lazy group is pending **[verified]** (P1)

`TypeRegistry._lazy_names` (`registry/type.py:158-161`) reads `key.__mro__`. `has`, `get`, `__getitem__`,
`unregister`, `__contains__` and `resolve` call it **before** the key is validated, so any non-type
raises as soon as the registry has an unloaded group. All default registries have one.

```python
from coola.equality.tester import get_default_registry

registry = get_default_registry()
"abc" in registry  # AttributeError: 'str' object has no attribute '__mro__'
registry.has(None)  # AttributeError
registry.find("abc")  # AttributeError (it already raised it before #2231)
```

Before #2231 `"abc" in registry` and `has("abc")` were `False` and `get("abc")` returned the default;
those are the regressions. `register` still raises the intended `TypeError` (#2162) because it validates
first. Fix: `_lazy_names` returns `()` for a non-type (`getattr(key, "__mro__", ())`), so the lookups
fall through to `key in self._state` (`False`), and make `resolve`/`find` of a non-type raise the
documented `TypeError` instead of the `AttributeError`. Add the regression test in
`tests/unit/registry/test_type.py` **and** one that goes through a default registry with a pending
group (the lazy tests added in #2231 never look up a non-type, which is why the suite stayed green).

### 12.2 Reading a registry loads every lazy group **[verified]** (P2)

`repr`, `str`, `len`, iteration, `items`, `keys`, `values` and `equal` call `_load_lazy_all`.

```python
from coola.equality.tester import get_default_registry
import sys

repr(get_default_registry())
[m for m in ("torch", "pandas", "polars", "pyarrow", "xarray", "jax") if m in sys.modules]
# all six
```

So printing a registry in a debugger, a log line or an error message imports JAX, torch, pandas, ...,
which is the cost #2231 removed. Fix: `_display_fields` of `BaseTypeDispatchRegistry` should show the
loaded entries and a placeholder for the pending groups (`<not loaded: jax, torch>`), without loading.
`len` can keep loading (it must be exact), but `repr` should not.

### 12.3 `BaseRegistry.equal` holds both locks while it runs the comparison **[reading]** (P3, _carried over_ from 8.19)

`registry/base.py:243-244` still runs `objects_are_equal(self._state, other._state)` under
`first._lock, second._lock`. PR #2195 (snapshot, release, then compare) was never merged. The id-based
lock order avoids the A/B deadlock, but a value whose `equal`/`__eq__` takes another registry's lock
(for example a tester that holds a registry) can still deadlock, and a slow comparison blocks all
registrations. Snapshot with `_get_snapshot()`, release, compare.

### 12.4 `HasherRegistry` fallbacks **[verified]** (P2)

Types that are handled by a fallback (every dataclass, every class with a `hash` method) take a
**failing lookup per object**: `find` raises `TypeNotRegisteredError`, whose message is built eagerly
(MRO, a `Counter`, a sorted list of all registered type names), then `find_hasher` catches it and walks
the predicates.

```python
@dataclass
class P:
    x: int


hash_object([P(i) for i in range(20000)])  # 0.33 s: 16 µs per object, 70% in find_hasher
hash_object(list(range(20000)))  # 0.034 s
```

Related defects in the same code (`hashing/registry.py:82-122`, `BaseTypeDispatchRegistry.equal`):

- `equal` compares only `_state`, so two registries with different fallbacks are equal **[verified]**.
- `clear()` does not reset `_fallbacks`; there is no `unregister_fallback`.
- `register_fallback` does `self._fallbacks = (*self._fallbacks, ...)` without the lock: two threads can
  lose one registration **[reading]**.
- The equality registry solves the same problem differently (`DataclassMarker` plus
  `_has_generated_eq`). Two mechanisms for "a type that cannot be registered one by one".

Fix: cache the fallback result per type (put the resolved type in the `TypeRegistry` cache, or keep a
small `WeakKeyDictionary`), build the error message lazily, lock the writes, include the fallbacks in
`equal`. Longer term: one `register_fallback(predicate, value)` in `BaseTypeDispatchRegistry` used by
both the equality and the hashing registries (and the dataclass summarizer of 15.3).

## 13. `equality`

### 13.1 No tester for pydantic models, and `==` can raise inside `ObjectEqualHandler` **[verified]** (P2)

Hashing, display and `to_jsonable` have pydantic support; equality does not, so a model is compared
with its own `__eq__`:

```python
class M(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    x: np.ndarray


objects_are_equal(M(x=np.ones(2)), M(x=np.ones(2)))  # ValueError: truth value of an array ... ambiguous
```

Two defects:

1. `ObjectEqualHandler.handle` (`handler/native.py:122-135`) wraps only `bool(result)` in the `try`. The
   `actual == expected` on the line before is outside it, so the documented "`TypeError: register an
equality tester`" does not cover a `==` that raises (pydantic with an array field, SciPy sparse
   gives the documented `TypeError` only because its `==` returns a matrix). Move the `==` into the
   `try` and re-raise as `TypeError` from the cause.
2. Even when `==` works, `equal_nan` and the tolerances are ignored for float fields:
   `objects_are_allclose(P(x=[1.0]), P(x=[1.0 + 1e-9]))` is `False`, and `equal_nan=True` does not
   make two `P(x=[nan])` equal. Add a `PydanticModelEqualityTester` (same type, then `model_fields`
   plus `model_extra`, each value through the registry; private attributes optional), registered
   lazily like the other library testers. A generic opt-in `ObjectAttributesEqualityTester`
   (`vars()`/`__slots__`) would cover plain classes and attrs classes the same way the dataclass tester
   does (`Foo(1)` and `Foo(1)` are `False` today, and so are two equal exceptions).

### 13.2 `objects_are_allclose` is asymmetric for NumPy and torch **[verified]** (P2)

`np.allclose` and `torch.allclose` test `|a - b| <= atol + rtol * |b|`: only the expected value scales
the tolerance. Scalars (`math.isclose`), pandas and polars use the larger of the two.

```python
kw = dict(rtol=0.01, atol=0.0)
objects_are_allclose(100.0, 101.005, **kw), objects_are_allclose(101.005, 100.0, **kw)  # True, True
objects_are_allclose(np.array([100.0]), np.array([101.005]), **kw)  # True
objects_are_allclose(np.array([101.005]), np.array([100.0]), **kw)  # False
```

So `objects_are_allclose(a, b)` can differ from `objects_are_allclose(b, a)`, and a value can pass as a
scalar and fail as a one-element array. For a library whose purpose is "are these two equal", symmetry
is expected. Fix in `handler/numpy.py:101`, `handler/torch.py:138` and `handler/jax.py:94`: use the
`math.isclose` rule that the scalar path uses, `|a - b| <= max(rtol * max(|a|, |b|), atol)`, with the
`inf`/`nan` rules of `allclose`, so that the array, scalar and DataFrame paths agree. Or document the
asymmetry in the "Tolerance" column of `docs/docs/uguide/equality.md` and in the docstrings.

### 13.3 Requested tolerance or `equal_nan` is silently not applied to some types **[verified]** (P3, documented)

`docs/docs/uguide/equality.md:813-819` documents it, but the user still gets `False` with no hint:

```python
objects_are_allclose(Decimal("1.0"), Decimal("1.00001"), atol=1e-3)  # False
objects_are_allclose(Fraction(1, 3), Fraction(333333, 1000000), atol=1e-3)  # False
objects_are_allclose(timedelta(seconds=1), timedelta(seconds=1.0001), atol=1e-3)  # False
objects_are_allclose({1.0}, {1.0 + 1e-9})  # False
objects_are_equal(np.datetime64("NaT"), np.datetime64("NaT"), equal_nan=True)  # False (the array version is True)
```

The pyarrow handler already has the right mechanism (`_warn_ignored` with `config._warned`, once per
setting). Options: (a) register tolerant testers for `Decimal`, `Fraction`, `timedelta`/`datetime`
(`abs(a - b) <= atol`) and treat NumPy `datetime64`/`timedelta64` NaT scalars like the arrays;
(b) make `DefaultEqualityTester` warn once when a tolerance or `equal_nan` was requested and the type is
not an exact-comparison type. Move the shared helper out of `handler/pyarrow.py` (the comment in
`config.py:95` refers to a `warn_ignored_setting` that does not exist).

### 13.4 `compare` and `assert_objects_equal` raise for a field with a failing `__str__` **[verified]** (P2)

`DataclassSameFieldsHandler`, `SameAttributeHandler` and `SameDataHandler` pass the raw objects as `%s`
arguments to `report_difference`. With a `reporter` (always the case in `compare`) the message is
formatted eagerly with `msg % args`:

```python
compare(D(Bad(1)), D(Bad(2)))  # ZeroDivisionError from Bad.__str__
```

With the logging path the error goes to logging's `handleError` ("--- Logging error ---" on stderr).
`format_value_difference` already uses `safe_str`; the three handlers should do the same (and truncate:
`SameAttributeHandler("coords.variables")` and a large `.data` are formatted in full). 5.12 of round 3
fixed the `repr` of the roots in `compare`, not these.

### 13.5 `ComparisonResult.actual` and `.expected` are unbounded **[verified]** (P3)

`compare` stores `safe_repr(actual)` and `safe_repr(expected)` for every failing comparison. For two
lists of 2M ints that is two strings of 16.9 MB each, and `format_message` (so `assert_objects_equal`)
does not use them. Truncate (`safe_repr(obj, max_chars=...)` with a `...` suffix, 1000 by default) or
make them lazy cached properties.

### 13.6 Container edge cases **[verified]** (P3)

- `dict_values` of equal dicts are `False` (the type has no `__eq__`, only identity):
  `objects_are_equal({"a": d.values()}, {"a": dict(d).values()})`. Register it with the sequence tester
  (order is the dict order) or document it.
- `Counter({"a": 0})` and `Counter()` are `==` since Python 3.10 but `False` here (the length handler
  runs first). Say so in the mapping section or register a `Counter` tester.
- `torch.nested` tensors raise `RuntimeError: NestedTensorImpl doesn't support sizes` (same family as
  the sparse and meta tensors fixed in #2073); NumPy structured arrays ignore `equal_nan`; SciPy
  sparse matrices give the clear `TypeError` of 13.1 and could simply get a tester (`(a != b).nnz == 0`
  plus format and shape; carried over from 5.2).

### 13.7 Sequences that differ always take the slow path **[verified]** (P3)

`SequenceSameValuesHandler` (`handler/sequence.py:84-90`) returns early only when the fast path says
"equal". When the items have the same exact types but different values, it falls through to the
per-item loop (0.8 µs per item): `objects_are_equal(big, big2)` for two lists of 2M ints that differ in
the last item takes 1.57 s, almost all of it spent re-proving that `actual != expected`. If the fast
path conditions hold and `actual != expected`, the answer is `False` immediately; the loop is needed
only to locate the index when `show_difference` is on.

## 14. `hashing`

### 14.1 NumPy structured dtypes collide **[verified]** (P1)

`NumpyArrayHasher` uses `data.dtype.str`, which is `'|V8'` for every 8-byte structured dtype:

```python
hash_object(np.zeros(2, dtype=[("a", "f8")])) == hash_object(np.zeros(2, dtype=[("b", "f8")]))  # True
hash_object(np.zeros(2, dtype=[("a", "f8")])) == hash_object(np.zeros(2, dtype=[("a", "i8")]))  # True
```

`objects_are_equal` says these arrays are different (different dtype). A content hash that is equal for
values that are not equal breaks `generate_stable_content_id`. Fix: for `dtype.kind == "V"` use
`str(data.dtype.descr)` (field names, formats, offsets) in `hash_array_content`, and normalise `-0.0` and
NaN inside the float fields. Add a Hypothesis strategy for structured dtypes (18.2).

### 14.2 pandas hashing ignores metadata that equality compares **[verified]** (P1)

`PandasSeriesHasher`/`PandasDataFrameHasher` hash `index.to_numpy()` and the column values. They drop:

| Dropped                        | `objects_are_equal` | `hash_object` |
| ------------------------------ | ------------------- | ------------- |
| `df.index.name` (`x` vs `y`)   | different           | **same**      |
| `df.columns.name` (`x` vs `y`) | different           | **same**      |
| `Categorical` `categories`     | different           | **same**      |
| `Categorical` `ordered`        | different           | **same**      |

(`df.attrs` is ignored by both, which is consistent: equality is the reference.) Fix: hash
`index.names`, `columns.names`, the index dtype, and for categoricals the categories and the `ordered`
flag. For a `MultiIndex` hash `index.to_frame(index=False)` instead of the object array of tuples.

### 14.3 NaN bit patterns change the hash of arrays and tensors **[verified]** (P2)

`-0.0` is normalised (`data + 0`), NaN is not. Floats and lists are canonical (`repr` gives `'nan'`),
arrays and tensors hash the raw bytes:

```python
hash_object(np.array([np.nan])) == hash_object(np.array([-np.nan]))  # False
hash_object(float("nan")) == hash_object(-float("nan"))  # True
objects_are_equal(np.array([np.nan]), np.array([-np.nan]), equal_nan=True)  # True
```

`np.inf - np.inf` produces the negative quiet NaN on x86-64 (default NaN of SSE, from the architecture
manual; not run here) and the positive one on Apple silicon (the probe here printed `000000000000f87f`
for both), so the same computation can give different IDs on different CPUs **[reading]**. The `-np.nan`
repro above is enough to show the difference on any machine. The same applies to `torch` (verified:
`False`), `jax`, and through them pandas and polars (they go through `to_numpy`). Fix: `np.where(np.isnan(data), np.nan, data)` for kinds `f` and `c` next to the `-0.0`
normalisation (and `torch.where(tensor != tensor, nan, tensor)`).

### 14.4 Types supported by equality but not by hashing, and wrong exceptions **[verified]** (P2)

From the support matrix (`TypeNotRegisteredError` unless stated):

- NumPy: `np.datetime64`, `np.void` scalars; `np.timedelta64` raises a raw `TypeError: int() argument
must be ... not 'datetime.timedelta'` because it is an `np.integer` and reaches `ReprHasher`.
- pandas: `Index`, `MultiIndex`, `DatetimeIndex`, `RangeIndex`, `Categorical`, extension arrays,
  `pd.NA`, `Interval`, `Period`; `pd.NaT` raises a bare `ValueError: NaTType does not support
utcoffset` (it is a `datetime` subclass, so it reaches `DatetimeHasher`).
- polars `LazyFrame`; pyarrow `Scalar`, `Schema`, `DataType`; `array.array`, `dict_keys`,
  `dict_values`, `dict_items`, `MappingProxyType`, `slice`, `Ellipsis`; torch `dtype`/`device`.

Equality has testers for the pandas `Index` and extension-array family (#2095), the hashing registry
does not. Add the hashers (index and extension arrays share the Series code path: hash `dtype`, `name(s)`
and the values), make `ReprHasher` skip `np.timedelta64`/`np.datetime64` (register a
`DatetimeHasher`-like hasher for them) and route `NaTType` before the `datetime` hasher. The parity
test of 18.1 keeps the two registries aligned.

### 14.5 Meta tensors and float8 tensors raise raw errors **[verified]** (P3)

`hash_object(torch.empty(2, device="meta"))` raises `NotImplementedError: Cannot copy out of meta tensor`
(equality and summary support meta tensors, #2073) and a float8 tensor raises `"add_stub" not
implemented for 'Float8_e4m3fn'` from the `tensor + 0` normalisation. Hash `(shape, dtype)` for meta
tensors, and for float8 view the bytes and normalise `-0.0` with `torch.where(t == 0, zeros, t)` or
skip the normalisation for those dtypes.

### 14.6 The "stable" identifiers have no golden vectors and no format version **[reading]** (P2)

`generate_stable_content_id` and `generate_stable_uuid5` promise stability in their names, and the guide
(`docs/docs/uguide/hashing.md:147`) says the values "can change between versions". During this cycle
they did change several times (#2083 new hashers, #2105 to #2126). No test pins a hash value of a
representative payload, so a refactor of a hasher changes persisted IDs without a failing test.

- Add `tests/unit/hashing/test_golden.py`: one fixed value per hasher family (int, float, str, bytes,
  list, dict, set, dataclass, datetime, Decimal, Enum, ndarray, tensor, DataFrame...). A change then
  needs a deliberate edit of the expected values.
- Add a public `HASH_FORMAT_VERSION` constant (and mention it in the release notes, 19.3). Optionally
  mix it into `generate_stable_uuid5`/`generate_stable_content_id` behind a `version=` argument so that
  callers can keep producing the old IDs.
- Either soften the names in the docs ("content ID, may change between minor versions") or promise
  stability from 1.x and enforce it with the golden test.

### 14.7 Equal dataclasses can hash differently **[verified]** (P2)

`DataclassHasher` hashes every field; `DataclassSameFieldsHandler` skips the fields with
`compare=False`, as `dataclass.__eq__` does:

```python
@dataclass
class D:
    a: int
    cache: int = field(default=0, compare=False)


D(1, 1) == D(1, 2), objects_are_equal(D(1, 1), D(1, 2)), hash_object(D(1, 1)) == hash_object(D(1, 2))
# True, True, False
```

This breaks the cross-module invariant "equal implies same hash" that `tests/properties/test_cross_module.py`
states. Hash only the fields with `compare=True` (and document `field(metadata=...)` if an opt-in for the
other fields is wanted), or document that `compare=False` fields are part of the hash.

## 15. `summary`

### 15.1 Collections at the depth limit call `str`/`repr` on the items without protection **[verified]** (P1)

`BaseCollectionSummarizer.summarize` (`summary/collection.py:103-112`) builds
`text = f"{preview!r} ..."` or `text = str(data)` for a collection that reaches `max_depth`. That runs the
`repr` of up to `max_items` **items**, outside the `safe_str` protection that #2071 added to
`DefaultSummarizer` (it only protects the final string):

```python
summarize([Bad()], max_depth=0)  # ZeroDivisionError (Bad.__repr__ raises)
summarize({"a": Bad()}, max_depth=0)  # ZeroDivisionError
summarize({Bad()}, max_depth=0)  # ZeroDivisionError
```

The same code builds the full string before truncating it to 1000 characters:
`summarize(["x" * 50_000_000] * 3, max_depth=0)` creates three 50 MB strings and takes 0.28 s.
`summarize` is the function used to build error messages, it must not raise. Fix: summarize each of the
first `limit` items with `registry.summarize(item, depth=depth + 1, max_depth=max_depth)` and join, or
at least use `safe_repr` per item with a per-item length bound. Add `Bad` items to the Hypothesis
strategy of `test_summarize_never_raises` (it currently generates only well-behaved literals).

### 15.2 `show_data=True` uses an unbounded, unprotected `repr` **[reading]** (P3)

`BaseMetadataSummarizer.summarize` (`summary/metadata.py:128-129`) returns `repr(data)`: no `safe_repr`,
no length bound (a `pa.Table` or a `LazyFrame` plan repr can be large). Use `safe_repr` and the
`DefaultSummarizer` truncation.

### 15.3 Dataclasses and pydantic models are summarized as one truncated string **[verified]** (P3)

Equality (`DataclassEqualityTester`) and hashing (`DataclassHasher`) see inside dataclasses, the
summary does not. The same `DataFrame` is summarized as `<class 'pandas.DataFrame'> | shape=(100, 1) |
columns=a: int64` inside a dict, but as the full multi-line pandas `repr` inside a dataclass, because
the dataclass goes to `DefaultSummarizer` (`str(config)`, cut at 1000 characters), and `max_depth` has
no effect on it:

```python
summarize(Config(name="x", data=np.arange(2000.0), frame=pd.DataFrame({"a": range(100)})), max_depth=3)
# Config(name='x', data=array([0.000e+00, ..., 1.999e+03], shape=(2000,)), frame=     a\n0    0\n1    1 ...
```

Add a `DataclassSummarizer` (field names and `registry.summarize` of the values, with the depth rules
of the collections, registered through the fallback of 12.4) and a pydantic one that reuses the secret
masking of `display.pydantic`.

## 16. `io`, `factory`, `random`

### 16.1 `str` paths are not accepted **[verified]** (P2)

```python
save_text("x", "out/a.txt")  # AttributeError: 'str' object has no attribute 'is_dir'
load_text("out/a.txt")  # AttributeError: 'str' object has no attribute 'open'
```

All the signatures say `Path`, but nearly every other path API of the library goes through
`sanitize_path` (`str | os.PathLike`), and callers pass strings. Convert with
`Path(os.fspath(path))` at the entry of `BaseFileSaver.save` and of each `load` (or raise a `TypeError`
that names the argument). Document the choice and test both.

### 16.2 The save lock costs 8x a plain write and starts a thread per call **[verified]** (P3)

`save_text("x", path)` takes 0.82 ms, a plain `write_text` 0.10 ms (300 files in a temporary directory
on macOS, so the figure is indicative). The difference
is the sidecar `.lock` file, the lock token, the heartbeat thread started and joined per call, the tmp
file and the `link`. It protects against two writers racing on one path, a case where "last writer wins
atomically" (the `replace`/`link` commit is already atomic) is often enough, and the protocol has known
holes (7.3 and 7.5 of round 3: the stale-lock window is narrowed, not closed). Consider
`save(..., lock=True)` (or a `lock=False` class option) so batch jobs that write thousands of distinct
files can skip it, and document the filesystems where `O_EXCL` locks are unreliable (NFS).

### 16.3 `resolve_saver`/`resolve_loader` execute the target before the type check **[verified]** (P2)

`resolve_object` builds the object with `factory(**config)` and only then checks
`isinstance(obj, cls)` (`factory/resolve.py:201-206`):

```python
resolve_saver({"_target_": "os.mkdir", "path": "/tmp/marker"})
# TypeError: Received object is not a BaseSaver instance (received: <class 'NoneType'>)
os.path.isdir("/tmp/marker")  # True: the call ran
```

The `Security:` notes document that `_target_` is code, and `allowed_prefixes` exists, so this is
hardening, not a new hole. But `resolve_loader`/`resolve_saver` know the only acceptable result type, so
they can refuse before calling: when the target is a class, check `issubclass(target, cls)`; when it is a
function, use the return annotation as `is_object_config` does (and refuse a function with no usable
annotation unless the caller opts in). Same for `is_object_config`, which imports the module of an
untrusted `_target_` just to answer a boolean.

### 16.4 Reading the torch RNG state may initialise CUDA **[lead]** (P3)

`TorchRandomManager.get_rng_state` calls `torch.cuda.get_rng_state_all()` (`random/torch.py:53`), which
calls `_lazy_init()` for each device, whereas `manual_seed` uses the deferred
`torch.cuda.manual_seed_all`. On a GPU machine, `random_seed(0)` (which saves the state first) can
initialise CUDA contexts, and in a forked worker that fails with "Cannot re-initialize CUDA in forked
subprocess". Not reproducible here (no GPU). Guard with `torch.cuda.is_initialized()` (an
uninitialised CUDA has no state to save) and add a test with the device functions patched.

### 16.5 `coola.random` imports NumPy and torch on first use **[verified]** (P3)

The first `manual_seed(1)` takes about 0.9 s here because the default registry imports both libraries
when they are installed (the comment in `random/interface.py:218` explains why: a library imported later
must still be seeded). Say so in the docstrings of `manual_seed`/`random_seed` so that nobody
discovers it in a test run, or offer `register_managers` with a lazy manager that seeds on import.

## 17. `utils`, `nested` and small API issues

### 17.1 Two "to JSON" functions, one still not strict-JSON safe **[verified]** (P3, _carried over_ from 1.5)

`coola.utils.to_jsonable` converts one level (it leaves lists and dicts as they are, by design) and
`coola.nested.convert_to_jsonable` is the recursive one. Names and places are easy to confuse. What is
still not converted by `convert_to_jsonable`: non-string keys (`{(1, 2): 1}` raises `TypeError` in
`json.dumps`, `{1: 2}` silently becomes `"1"`) and `nan`/`inf` (`allow_nan=False` raises). `save_json`
still writes the non-standard token `NaN` and turns int keys into strings on the round trip. Decide the
contract (strict JSON or "Python `json` module output") and state it in the three docstrings.

### 17.2 Dead or stale code **[verified]** (P3)

- `utils/lru.py`: `LRUCache` is used nowhere in `src` and its docstring says it is "used by
  `TypeRegistry`", which now uses a `WeakKeyDictionary`. Remove it or document it as standalone (it is
  not thread-safe, the docstring says so).
- `utils/bloom_filter.py`: the only operation is `add_and_check`, so there is no read-only query,
  no `__contains__`, no size/fill information; not thread-safe. Round 2 decided to keep it; if it
  stays, give it `__contains__`/`add` and document the thread-safety.
- `equality/tester/jax.py:80-95`: `get_array_impl_class()` creates an array (`jnp.ones(1)`) to get the
  concrete class, the pattern #2198 removed from the summary. It runs only when a JAX array is already
  being compared, so the backend is initialised anyway: cosmetic, but `jnp.ndarray` (`jax.Array`) is in
  the MRO of the concrete class, as the summary registration already relies on, so it is redundant.
- `equality/config.py:95` refers to `warn_ignored_setting`, which does not exist (13.3).

### 17.3 `import coola` spends most of its time on `importlib.metadata` **[verified]** (P3)

`coola/__init__.py` calls `version(__name__)` at import to fill `__version__`: `-X importtime` shows
`importlib.metadata` at 18 ms of the 25 ms of `import coola`, and it pulls `email`, `zipfile` and `json`.
Every subpackage pays it. Use a module-level `__getattr__("__version__")` (PEP 562) that reads the
metadata on first access; `lazy_exports` already shows the pattern.

### 17.4 Eleven copies of the same availability module **[reading]** (P3)

`utils/imports/{colorlog,jax,numpy,pandas,polars,pyarrow,pydantic,rich,torch,xarray,packaging}.py` are
97 to 102 lines each with the same four functions (`check_X`, `is_X_available`, `X_available`,
`raise_X_missing_error`), 44 exported names, plus `utils/fallback/*` with the same shape. Adding a
backend (see `docs/docs/dev/add_backend.md`) means copying a file and editing three strings.
A small factory (`make_package_helpers("numpy", install="numpy")` returning the four callables, still
declared explicitly in each module for the type checker) would cut several hundred lines and remove
the drift risk (7.21 of round 3 was one such drift).

### 17.5 `KeyFilterTransformer` legacy mode **[reading]** (P3, _carried over_ from 9.17)

`recursive/key.py:140-157` chooses the function to apply to a value by sniffing its type
(`str` → `transform_func`, container → `func`, anything else → `transform_func`). It is fragile and
only exists for the legacy `transform_func` mode. Deprecate the mode (a `DeprecationWarning` now, removal
in the next breaking release) and keep the `key_predicate` mode only.

### 17.6 `to_flat_dict`/`from_flat_dict` do not round-trip keys that contain the separator **[verified]** (P3)

`to_flat_dict({"a.b": 1, "c": {"d": 2}})` gives `{"a.b": 1, "c.d": 2}` and `from_flat_dict` of that gives
`{"a": {"b": 1}, "c": {"d": 2}}`. Add an `escape` option (or reject keys that contain the separator in
`to_flat_dict` with a clear error) and document the rule next to `restore_lists`.

### 17.7 `coola.testing` needs pytest, which is in no extra **[reading]** (P3)

`coola/testing/fixtures.py` imports `pytest` at the top, and `pytest` is only in the `dev` group. Add a
`testing = ["pytest>=..."]` extra (the autoextras hook then lists it in `all`) or import lazily and raise
a `MissingPackageError`.

## 18. Tests, CI and tooling

### 18.1 No test keeps `equality`, `hashing` and `summary` aligned **[verified]** (P2)

Most of sections 13, 14 and 15 are the same shape: one module supports a type, another does not. The
support-matrix script of this review (about 100 types × `equal`, `hash`, `summarize`, `dfs_iterate`,
`recursive_apply`) found 41 rows with a failure. `dfs_iterate`, `recursive_apply` and `summarize` pass
for all of them; `equal` fails only for SciPy sparse matrices, `pl.Expr` and objects without value
semantics (`object()`, iterators, exceptions); `hash_object` is the gap (all 41 rows). Turn the script into
`tests/integration/test_support_matrix.py` with a table of `(object factory, supported_by)` and
`xfail(strict=True)` for the known gaps, so each fixed gap flips the xfail and each new type must be
added to the table. Generate the "supported types" table of the docs (`equality.md`, `hashing.md`,
`summary.md`) from the same table.

### 18.2 The property tests do not generate the classes of defect found here (P2)

`tests/properties/strategies.py` builds nested containers of scalars, and `test_cross_module.py` states
the right invariants (equal implies same hash, `summarize` never raises, strict JSON). The defects of
14.1 to 14.3, 14.7 and 15.1 are outside the generated space. Add, to the library-type strategies:
`-0.0`, `-nan`, structured dtypes, DataFrames with index/column names and categoricals, `NaT`,
dataclasses with `compare=False` fields, and items whose `__repr__`/`__str__` raise. The invariants
catch each of them as is.

### 18.3 Local imports in tests **[verified]** (P3)

`CLAUDE.md` forbids imports inside test bodies. An AST scan finds 8 in 3 files:
`tests/unit/hashing/test_interface.py:645` (5), `tests/benchmarks/conftest.py` (2),
`tests/unit/hashing/test_repr.py:235` (1). Move them to the top (or into a fixture for the benchmark
conftest, where the import may be deliberately delayed: say so in a comment). The whole `PL` family
is in the `tests/**` per-file ignores, which hides `PLC0415` (import outside top level); selecting
`PLC0415` for the tests, with a per-file ignore for `tests/benchmarks/conftest.py`, would make the
`CLAUDE.md` convention enforceable.

### 18.4 Smaller points (P3)

- Coverage is 99% but `[tool.coverage.report]` has no `fail_under`; add `fail_under = 99` (or 98) so
  the number cannot drop silently. A mutation-testing run on `equality/handler` is still not done
  (carried over from the earlier reviews).
- `[tool.ty.src] include = ["src"]`: the tests are not type-checked. Adding `tests/unit` as a
  non-blocking CI job would show how much of the fixtures typing is missing.
- A `pytest` plugin in `coola.testing` that implements `pytest_assertrepr_compare` for nested data
  would make a plain `assert actual == expected` print the path of `compare()` (`['a'][2]`) without
  changing the tests. Opt-in through a `pytest11` entry point. This is a feature, not a fix.

## 19. Packaging and documentation

### 19.1 Classifiers and Python 3.15 (P3, _carried over_ from 2.3)

`Development Status :: 5 - Production/Stable` is still declared for `1.2.0a2`, and Python 3.15 is in the
classifiers. Use `4 - Beta` (or `3 - Alpha`) until the release, and keep 3.15 only if it is in the
required CI matrix (`lib-get-test-matrix.yaml` takes it from an external action, so the repo itself
cannot show it).

### 19.2 No changelog or migration notes since 1.0 (P2)

The only migration page is `docs/docs/migration/v0.11_to_v1.0.md`, and the changelog points to GitHub
releases. This cycle contains changes that users can notice: hash values (14.6), the default `max_depth`
of 1000 lowered to 100 (5.13), `hash_object`/`generate_stable_*` options keyword-only (6.12), mapping
keys compared with their type (5.4), `get_default_generator` removed from `__all__` (6.20),
`TypeRegistry` rejecting non-type keys (8.2). Add a `CHANGELOG.md` (or a "What changed in 1.2" page) that
lists them under "Behaviour changes", and mention the file in `CONTRIBUTING.md`; the PR template of
`CLAUDE.md` ("any behaviour change") already collects the content.

### 19.3 Docstring slips (P3)

- `objects_are_equal` repeats the tolerance paragraph of `objects_are_allclose` ("so `True` and `False`
  are close with `atol=1`"), which does not apply to an exact comparison; its `Raises` says
  `RecursionError`, the class is `MaxDepthExceededError` (a subclass, and the Python `RecursionError`
  is wrapped too, `equality/interface.py:25-32`).
- README overview: the first paragraph still says the library "makes it easy to compare complex and
  nested data structures", while the feature list (hashing, summaries, identifiers, I/O, random) is
  half of the package. `pyproject.toml` was updated in #2080, the README was not.

## 20. Status of round 3

Checked against the git history up to `0287a61a` and by re-running the repros marked in this file.
Round 3 contained the items 1.1 to 10.18. **All but the ones below were merged** (PRs #2070 to #2217,
see `git log --grep` for the item numbers), including every "PR open" row of the round-3 table.

| Round-3 ID                              | Status now    | Notes                                                                                                                                                                                                                                                                   |
| --------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.3                                     | Open (doc)    | The masked-array rule (data under the mask is compared) is still not documented in `equality.md`.                                                                                                                                                                       |
| 1.5                                     | Partly        | Tuple keys, `nan`/`inf` and int keys still not strict-JSON: see 17.1.                                                                                                                                                                                                   |
| 2.3                                     | Partly        | See 19.1.                                                                                                                                                                                                                                                               |
| 2.4                                     | Decided       | `utils`/`identifier` stay (decision in `src-coola-review.md` 1.1). 17.2 lists the unused pieces.                                                                                                                                                                        |
| 2.5                                     | Already there | `coola.random.random_seed(seed)` is the "seed everything and restore" context manager; only the first-call cost is open (16.5).                                                                                                                                         |
| 3                                       | Partly        | Windows is a non-goal (7.20); `fail_under` and mutation testing: 18.4.                                                                                                                                                                                                  |
| 5.2                                     | Partly        | `pd.Index` and extension arrays done; SciPy sparse still raises the `TypeError` of 13.1/13.6.                                                                                                                                                                           |
| 5.3, 5.4, 5.6, 5.9, 5.10, 5.12          | Fixed         | Verified fixed on `main` (custom `__eq__` of dataclasses respected, keys compared with their type, documented identity shortcut, `None` key sentinel, `HandlerCycleError` for `a -> b -> a`, `safe_repr`). NaN keys never match, even with `equal_nan=True` (P3, left). |
| 5.15                                    | Partly        | The fast path for lists of floats and the missing terminal handler check are still open; see 13.7 for the sequence fast path.                                                                                                                                           |
| 6.10, 6.18                              | Documented    | Module/class name in hashes, parity leak of the obfuscated IDs: documentation only, by decision.                                                                                                                                                                        |
| 6.21                                    | Partly        | `PureWindowsPath("a/b")` and `PurePosixPath("a/b")` still hash the same (hashed as POSIX, probably intended: add one sentence to `hashing.md`).                                                                                                                         |
| 7.3, 7.5                                | Partly        | Dead-owner detection and the renewed-lock restore merged; no kernel locks, timeout not configurable (16.2).                                                                                                                                                             |
| 7.24                                    | Partly        | `float()` of ints above 2**53 in `stats.py` and `fsync` on `BaseSaver.save` unchanged.                                                                                                                                                                                  |
| 8.4                                     | Mostly        | A generic that contains a secret (`tuple[SecretStr, int]`) still drops the whole field.                                                                                                                                                                                 |
| 8.9                                     | Mostly        | `max_items < 0` still previews 5 items (deliberate).                                                                                                                                                                                                                    |
| 8.15                                    | Mostly        | The `Summarizer` protocol still has no `equal`.                                                                                                                                                                                                                         |
| 8.17                                    | Fixed         | Lazy registries (#2231). New problems from it: 12.1, 12.2.                                                                                                                                                                                                              |
| 8.19                                    | **Open**      | PR #2195 was not merged: 12.3.                                                                                                                                                                                                                                          |
| 9.14                                    | Mostly        | `auto_reducer()` still prefers torch over NumPy for plain lists (the import is the cost).                                                                                                                                                                               |
| 9.17                                    | Open (design) | 17.5.                                                                                                                                                                                                                                                                   |
| 10.6                                    | Partly        | The Markdown and Makefile linters still fail on the current tree and are not in CI.                                                                                                                                                                                     |
| 10.9, 10.11, 10.12, 10.13, 10.15, 10.17 | Partly        | As in the round-3 table (external `invoke_tasklib` globs, `equality.py` example checks only the exit code, test mirrors, no `io`/`utils` property tests, wheel-less matrix cells, overlap of the dependency bots). Not re-checked.                                      |

## 21. Suggested order

1. **Regression and crashes (one small PR each, with the test that would have caught it):** 12.1
   (non-type keys), 15.1 (`summarize` at the depth limit), 13.4 (`compare` with a failing `__str__`),
   13.1 first half (`==` that raises).
2. **Collisions and invariants of the hashing module:** 14.1, 14.2, 14.7, 14.3, then 14.6 (golden
   vectors, before changing more hashes, so the changes of 14.1 to 14.3 are visible in the diff).
3. **Parity:** 18.1 (the matrix test), then 14.4/14.5 (hashers for what equality already supports) and
   13.1 second half (pydantic tester), 15.3 (dataclass summarizer).
4. **Consistency:** 13.2 (symmetric tolerance), 16.1 (`str` paths), 16.3 (check before calling),
   12.2 (lazy `repr`), 12.4 (fallback cache).
5. **Cleanup and docs:** 12.3, 13.3, 13.5 to 13.7, 17.x, 18.2 to 18.4, 19.x.
