# Codebase review: suggestions (round 3)

This file **replaces the previous `review/suggestions.md`** (417 lines, items 1.1 to 2.10 handled).
That version is in git history (last touched by `55086b7f`). Nothing below repeats an item of that
review, `src-coola-review.md`, `noqa-review.md` or `python-311-modernization.md`, except where
an item is explicitly marked _still open_.

## How this review was done

Two passes on the checkout (`.venv`, Python 3.14, NumPy, torch, pandas, polars, pyarrow, xarray, JAX,
pydantic installed):

1. **Part A (sections 1 to 4)**: my own probes of `utils`, `io`, `identifier`, `nested`, `random`,
   `hashing`, `summary` and the third-party equality testers.
2. **Part B (sections 5 to 10)**: a full read-through, in parallel, of every package. One reviewer per
   group read every file in full: `equality`, `hashing` + `identifier`, `io` + `utils`, `registry` +
   `summary` + `factory` + `validation` + `display` + `testing`, `iterator` + `recursive` + `nested` +
   `reducer` + `random`, and everything outside `src/coola` (tests, CI, docs, tooling). The tests (about
   50k lines) were surveyed and sampled, not read line by line.

Items marked **[verified]** were reproduced; the others are leads from reading the code. Baseline:
`pytest tests` gives 8521 passed, 39 skipped, 3 xfailed, and `ruff check .` is clean. The review changed
no source file. Where Part B repeats a Part A item it is merged here (for example the big-integer crash
appears in both `summarize` and `hash_object`).

Priority: **P1** wrong result or leak on realistic input, **P2** inconsistent or surprising behaviour,
**P3** cleanup.

## Status (updated 2026-10-22)

Checked against the git history up to `5008f540`. "Fixed" means a merged commit addresses the item;
the PR number is given. Items not listed are **open**.

### Part A

| ID   | Status         | Notes                                                                                                                                                                                                   |
| ---- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1  | Fixed (#2070)  | Negative `show_first`/`show_last` rejected.                                                                                                                                                             |
| 1.2  | Fixed (#2071)  | Default summarizer survives `str`/`repr` failures.                                                                                                                                                      |
| 1.3  | Fixed (#2072)  | Object-dtype arrays compared element-wise (also 5.1). Masked-array note not documented.                                                                                                                 |
| 1.4  | Fixed (#2073)  | Sparse and meta tensors supported.                                                                                                                                                                      |
| 1.5  | Partly (#2074) | Sets and ranges converted; `save_json` limits documented. Non-string keys, `nan`/`inf`, `Path`, `datetime` still not converted by `convert_to_jsonable`.                                                |
| 1.6  | Fixed (#2075)  | Line endings preserved.                                                                                                                                                                                 |
| 1.7  | Fixed (#2076)  | Seed validated once before seeding.                                                                                                                                                                     |
| 1.8  | Fixed (#2077)  | Negative/non-finite durations; `str_indent` skips empty lines.                                                                                                                                          |
| 1.9  | Fixed (#2078)  | Compound suffix allow-list.                                                                                                                                                                             |
| 1.10 | Mostly (#2079) | Fixed: `verify_checksummed_id`, nano ID alphabet, `get_file_size`, `get_current_branch` (documented), `temp_env_vars` (documented), `check_env_vars` level. Not checked/open: `get_password` TTY check. |
| 2.1  | Fixed (#2083)  | Default hashers for NumPy, torch, JAX, pandas, polars, PyArrow, xarray.                                                                                                                                 |
| 2.2  | Fixed (#2081)  | PyArrow numeric columns honour `equal_nan`, `atol`, `rtol` (also 5.11 in part).                                                                                                                         |
| 2.3  | Partly (#2080) | Description and keywords updated. The 3.15 classifier and Development Status were not addressed.                                                                                                        |
| 2.4  | Open           | Scope of `utils`.                                                                                                                                                                                       |
| 2.5  | Open           | `seed_everything` context manager.                                                                                                                                                                      |
| 3    | Partly         | Cross-module property tests added (#2084); `make clean` added (#2085). Regression tests exist for the fixed items; Windows runner, mutation testing, coverage `fail_under` still open.                  |

### Part B

| ID   | Status                          | Notes                                                                                                                                                                                                 |
| ---- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1  | Fixed (#2072)                   | Same fix as 1.3.                                                                                                                                                                                      |
| 5.2  | Fixed (#2095)                   | Testers for pandas `Index` and extension arrays. Scipy sparse not checked.                                                                                                                            |
| 5.5  | Fixed (#2096)                   | Nested polars NaN detected.                                                                                                                                                                           |
| 5.7  | Documented (#2094)              | Scalar comparison rules documented, behaviour unchanged.                                                                                                                                              |
| 5.8  | Fixed (#2097, #2104)            | Dataclass fields and attributes in the `compare` path; unused `coola_path_kind` removed.                                                                                                              |
| 5.13 | Fixed (#2092)                   | Default `max_depth` lowered to 100.                                                                                                                                                                   |
| 5.14 | Fixed (#2091, #2102)            | Exact integer comparison with a tolerance; complex NaN parts compared separately.                                                                                                                     |
| 5.15 | Partly (#2098)                  | Duplicate validation, config comments and the sequence set per call cleaned up. Other points open.                                                                                                    |
| 6.1  | PR open (#2105)                 | Integers above the int-to-str limit hashed in hexadecimal (`int`, `Fraction`, `range`).                                                                                                               |
| 6.2  | PR open (#2106)                 | `DecimalHasher` canonical form built from `as_tuple()`.                                                                                                                                               |
| 6.3  | PR open (#2107)                 | Aware datetimes near `min`/`max` no longer overflow.                                                                                                                                                  |
| 6.4  | PR open (#2108)                 | Snowflake lock held during the next-millisecond wait.                                                                                                                                                 |
| 6.5  | PR open (#2109)                 | Default Snowflake and ObjectId generators use `LazySingleton`.                                                                                                                                        |
| 6.6  | PR open (#2111)                 | Explicit timestamp leaves the Snowflake state untouched (random sequence).                                                                                                                            |
| 6.7  | PR open (#2112)                 | Mapping entries sorted by the full key and value hash.                                                                                                                                                |
| 6.8  | PR open (#2113)                 | Enum mixins routed to `EnumHasher`.                                                                                                                                                                   |
| 6.9  | PR open (#2114)                 | `hash` method fallback requires the `SupportsHash` signature.                                                                                                                                         |
| 6.10 | PR open (#2115)                 | Documented only; module and class name stay in the hash.                                                                                                                                              |
| 6.11 | PR open (#2116)                 | `ReprHasher` fallback tag is `module.qualname` (builtins keep the bare name).                                                                                                                         |
| 6.12 | PR open (#2117)                 | `generate_stable_content_id` and `generate_stable_uuid5` options are keyword-only.                                                                                                                    |
| 6.13 | Partly, PR open (#2118)         | Cycle guard for nested pydantic models. A model inside a dataclass still uses the default `on_secret` (documented).                                                                                   |
| 6.14 | PR open (#2119)                 | `ValueError` instead of `RecursionError` for data nested too deeply.                                                                                                                                  |
| 6.15 | PR open (#2120)                 | ULID range validated; lowercase accepted.                                                                                                                                                             |
| 6.16 | PR open (#2121)                 | Non-ASCII input rejected by `verify_checksummed_id`. `sep` is still case-sensitive.                                                                                                                   |
| 6.17 | PR open (#2122)                 | `length` capped at 1024; `decode_obfuscated_id` input bounded.                                                                                                                                        |
| 6.18 | PR open (#2123)                 | Documented only (parity leak); no Feistel step.                                                                                                                                                       |
| 6.19 | PR open (#2124)                 | Shared length validation, checked before hashing; `memoryview` supported.                                                                                                                             |
| 6.20 | PR open (#2125)                 | `get_default_generator` removed from both `__all__`.                                                                                                                                                  |
| 6.21 | Partly, PR open (#2126)         | Prefix and generator result validated, `chars` tidied. Open: Windows/Posix path hash, duplicate registry export, `register_hashers` doctest.                                                          |
| 7.1  | PR open (#2132)                 | Temporary and lock names derived from a hash when the target name is long.                                                                                                                            |
| 7.10 | PR open (#2136)                 | `check_env_vars` treats an empty or blank value as not defined.                                                                                                                                       |
| 7.11 | PR open (#2139)                 | `sort_by_keys` falls back to (type name, key); used by the `sorted_keys` options.                                                                                                                     |
| 7.12 | PR open (#2140)                 | `to_tensor` calls `check_torch()` first.                                                                                                                                                              |
| 7.13 | Partly (#2141)                  | Dunder names and `_module` raise `AttributeError` without importing. `hasattr(proxy, "a")` of a missing package still raises `ModuleNotFoundError` on purpose; `copy.copy` is impossible for modules. |
| 7.14 | PR open (#2143)                 | `LazySingleton` uses a sentinel, so a `None` value is cached.                                                                                                                                         |
| 7.15 | Fixed (options) (#2150)         | Options `level` and `all_threads` added; defaults unchanged.                                                                                                                                          |
| 7.16 | PR open (#2145)                 | Host in a file URI and unknown `~user` raise `ValueError`; a missing path starts from its parent.                                                                                                     |
| 7.17 | PR open (#2144)                 | Unit chosen after rounding.                                                                                                                                                                           |
| 7.18 | PR open (#2142)                 | `TorchLoader.equal` applies the `weights_only` default.                                                                                                                                               |
| 7.19 | PR open (#2146)                 | `NotADirectoryError` when the parent is a file.                                                                                                                                                       |
| 7.2  | PR open (#2133)                 | `unified_diff` marks lines without a trailing newline like `git diff`.                                                                                                                                |
| 7.20 | Won't fix (#2156; #2152 closed) | Windows is not a supported OS for coola; PR #2152 closed.                                                                                                                                             |
| 7.21 | PR open (#2148)                 | Error `name` set and message tidied; `package_available` catches `ValueError`/`ImportError`.                                                                                                          |
| 7.22 | PR open (#2149)                 | Invalid versions give `None`; cache cleared by `clear_availability_caches`.                                                                                                                           |
| 7.23 | Mostly (#2151)                  | The listed examples fixed. Other doctests of `utils` still fail under pytest's doctest runner (they pass with xdoctest).                                                                              |
| 7.24 | Mostly (#2155)                  | Fixed: bfloat16 `to_array`, resolved lock key, docs. Not changed: >2**53 ints in `stats.py`, `fsync` on `BaseSaver.save`.                                                                             |
| 7.3  | Partly (#2153)                  | Dead owner (same host) detected from the `host:pid:uuid` token and the lock broken at once. Timeout still not configurable; no kernel locks.                                                          |
| 7.4  | PR open (#2147)                 | Mode copied from the replaced file; a symlink is followed.                                                                                                                                            |
| 7.5  | Partly (#2154)                  | Renamed lock re-checked and restored if it was renewed. The window is narrowed, not closed (no kernel locks).                                                                                         |
| 7.6  | PR open (#2135)                 | `is_git_repo` compares to `true`; the stderr of git is kept in the error.                                                                                                                             |
| 7.7  | PR open (#2134)                 | `to_date` returns the date part of a `datetime`.                                                                                                                                                      |
| 7.8  | PR open (#2138)                 | `ß` and similar letters mapped; non-ASCII-only text falls back to a hash.                                                                                                                             |
| 7.9  | PR open (#2137)                 | `datetime.time` becomes an ISO string and `timedelta` a number of seconds.                                                                                                                            |
| 8.1  | Fixed (#2161)                   | `PolarsLazyFrameSummarizer` shows `<unresolved schema: ExceptionType>` for an invalid plan.                                                                                                           |
| 8.2  | Fixed (#2162)                   | `TypeRegistry` raises `TypeError` for a non-type key (`register`, `register_many`, `initial_state`).                                                                                                  |
| 8.3  | Fixed (#2164)                   | `exclude_fields` warns only for names that are not fields of the model. Docstrings corrected.                                                                                                         |
| 8.4  | Mostly (#2163)                  | `SecretBytes` and `Secret` omitted like `SecretStr`. A generic with a secret still drops the whole field.                                                                                             |
| 8.5  | Mostly (#2165)                  | On a TTY `stream`, `datefmt` used; `handlers`, `filename`, `format`, `style` give plain config; `stderr=None` handled.                                                                                |
| 8.6  | Fixed (#2166)                   | "Does not exist" only for a missing module or attribute; a `str` `allowed_prefixes` is one prefix.                                                                                                    |
| 8.7  | Fixed (#2167)                   | `import_object("")` raises `ImportError`; `resolve_object` accepts a tuple; `is_object_config` robust to bad input.                                                                                   |
| 8.8  | Fixed (#2168)                   | `is_object_config` checks `get_origin(t) or t`, so `-> list[int]` matches `list`.                                                                                                                     |
| 8.9  | Partly (#2169)                  | Preview keeps the type (tuple, deque, set, frozenset, OrderedDict, Counter). `max_items < 0` still capped at 5 (deliberate, #27).                                                                     |
| 10.1 | Fixed (#2172)                     | `summary.md` prints stable `find_summarizer` lookups instead of the whole registry. `factory.md` updated for 8.6.                                                                                     |

All other items in sections 5 to 10 (including 7.x, 8.x, 9.x, 10.x) have no matching commit and are **open**.
The earlier "Suggested order" lists below are unchanged.

---

## 1. Correctness and robustness

### 1.1 `mask_secret` leaks the secret with negative arguments **[verified]** (P1) — **Fixed (#2070)**

`utils/secret.py` never validates `show_first` / `show_last`. Python slicing turns a negative value into
"all but the last N":

```python
mask_secret(
    "abcdefghijkl", show_first=-2
)  # 'abcdefghij**********ijkl'  (10 of 12 characters shown)
mask_secret("abcdefghijkl", show_last=-2)  # 'abc***********'
```

Reject negative values with `validate_ge(..., 0)` (the `validation` module is already used elsewhere in
`utils`). Also consider whether it is acceptable that the output length reveals the secret length for
short secrets (`'*' * len(secret)`); a fixed-width mask is the safer default for a function named
`mask_secret`.

### 1.2 `summarize` raises on values that cannot be converted to `str` **[verified]** (P1) — **Fixed (#2071)**

`summarize` is the function used to build error and log messages, so it should never be the thing that
fails.

```python
summarize(10**5000)  # ValueError: Exceeds the limit (4300 digits) ...
summarize(type("B", (), {"__repr__": lambda s: 1 / 0})())  # ZeroDivisionError
```

The default summarizer calls `str(data)` / `repr(data)` before truncating. Wrap that call in a
`try/except Exception` that falls back to `<ClassName object at 0x...>`, and truncate big integers by
`int.bit_length()` rather than converting them.

### 1.3 `equal_nan=True` is ignored for object-dtype NumPy arrays **[verified]** (P2) — **Fixed (#2072)**

```python
a = np.array([np.nan], dtype=object)
objects_are_equal(a, a.copy(), equal_nan=True)  # False
```

Pandas columns of `object` dtype that hold floats take the same path. Either compare object arrays
element-wise through `objects_are_equal` (so `equal_nan` and the nested testers apply), or warn like the
PyArrow tester does. Related: `np.ma.MaskedArray` with different data under the mask compares unequal
(`eq(ma([1, 2], mask=[0, 1]), ma([1, 9], mask=[0, 1]))` is `False`). That is defensible, but it is not
documented and the masked entries are not the user-visible values.

### 1.4 Torch sparse and `meta` tensors crash instead of comparing **[verified]** (P2) — **Fixed (#2073)**

```python
objects_are_equal(
    torch.eye(2).to_sparse(), torch.eye(2).to_sparse()
)  # NotImplementedError (aten::equal)
objects_are_equal(
    torch.empty(2, device="meta"), torch.empty(2, device="meta")
)  # NotImplementedError
```

`objects_are_equal` is documented as returning a `bool`, and 1.1 of the previous review made that a
contract. Handle `layout != torch.strided` (compare `to_dense()`, or indices and values) and
`device.type == "meta"` (compare shape and dtype only), or convert `NotImplementedError` into a clear
`TypeError` that names the tensor kind. `summarize` already handles meta tensors, so the support
is inconsistent across modules.

### 1.5 `convert_to_jsonable` does not produce JSON **[verified]** (P2) — **Partly fixed (#2074)**

The docstring says "JSON-compatible representation", but only dataclasses and pydantic models are
converted:

```python
json.dumps(convert_to_jsonable({"a": {1, 2}}))  # TypeError (set)
json.dumps(convert_to_jsonable({"a": range(2)}))  # TypeError (range)
json.dumps(convert_to_jsonable({(1, 2): 1}))  # TypeError (tuple key)
json.dumps(convert_to_jsonable({"a": float("nan")}), allow_nan=False)  # ValueError
```

Either rename or reword (it is "dataclass/pydantic to dict"), or finish the job: `set` and `frozenset`
to sorted lists, `range` to a list, non-string keys to `str`, `nan`/`inf` to `None` or a string, `Path`
to `str`, `datetime` to ISO format. The same applies to `save_json`: int keys silently become strings on
the round trip (`{1: 'a'}` loads as `{'1': 'a'}`) and `NaN` is written as the non-standard token `NaN`.
Pass `allow_nan=False` by default or document both.

### 1.6 Text round trip is not lossless **[verified]** (P2) — **Fixed (#2075)**

`save_text("a\r\nb\rc", p); load_text(p)` returns `'a\nb\nc'`: `TextLoader` opens the file in text mode with
universal newlines, so `\r\n` and `\r` are rewritten. On Windows `TextSaver` also translates `\n` to
`\r\n`. Open both with `newline=""`, or add a `newline` option and say so in the docstring. The same
applies to `JsonLoader`/`JsonSaver`, which are less affected because JSON escapes control characters.

### 1.7 `manual_seed` can leave the generators half-seeded **[verified]** (P2) — **Fixed (#2076)**

`manual_seed(2**70)` raises `ValueError: Overflow when unpacking long long` from the torch manager
_after_ `random` and NumPy were already seeded (NumPy uses `seed % 2**32`, so it accepts anything).
Two consequences: the global state is partially changed when the call fails, and
`seed` and `seed + 2**32` give the same NumPy stream without any warning.
Validate the seed once in `manual_seed` (`0 <= seed < 2**63`, an `int`) before calling the managers, and
state the range in the docstring. Negative seeds are currently accepted by two of the three backends.

### 1.8 `str_time_human` on negative, NaN and infinite durations **[verified]** (P3) — **Fixed (#2077)**

```python
str_time_human(-5)  # '-1 day, 23:59:55'
str_time_human(float("nan"))  # ValueError: cannot convert float NaN to integer
str_time_human(float("inf"))  # OverflowError
```

The first output is the `timedelta` default and is misleading; the other two are raw conversion errors.
Handle the sign explicitly (`'-0:00:05'`) and raise a message that names the argument, like
`str_human_byte_size` and `make_bar` do. Also `str_indent("a\nb\n", 2)` returns `'a\n  b\n  '`, which
indents the empty last line and leaves trailing whitespace; skip empty lines.

### 1.9 `add_uuid_suffix` treats dots in the stem as an extension **[verified]** (P3) — **Fixed (#2078)**

`add_uuid_suffix(Path("v1.2.model"))` gives `v1-<uuid>.2.model` because `''.join(path.suffixes)` takes
every dotted part. That is deliberate for `.tar.gz`, but a version number in a file name is common.
Limit the compound suffix to a known allow-list (`.tar.gz`, `.tar.bz2`, `.tar.xz`, ...) and use
`path.suffix` otherwise.

### 1.10 Smaller correctness leads (P3) — **Mostly fixed (#2079)**

- `verify_checksummed_id(None)` raises `AttributeError`; a verifier should return `False` for non-`str`
  input (or raise a `TypeError`).
- `generate_nano_id(5, alphabet="a")` returns `'aaaaa'`: an alphabet of one character gives zero
  entropy. Require at least 2 characters (ideally more), or at least document it.
- `get_file_size(dir)` counts symlinked files twice and raises `FileNotFoundError` if a file is deleted
  between `rglob` and `stat`. Use `os.scandir` with `follow_symlinks=False`, and skip vanished entries.
- `get_current_branch()` returns the literal `HEAD` on a detached checkout. Say so in the docstring or
  return `None`.
- `get_password` requires `stdin` and `stderr` to be TTYs, but `getpass` reads from `/dev/tty`
  and works when `stdin` is piped. The check rejects usable setups (for example `cmd | python script.py`).
- `temp_env_vars` is not thread-safe (it mutates `os.environ`). One sentence in the docstring is enough.
- `check_env_vars` logs absent variables at `WARNING` even when `raise_on_missing=False` and the caller
  treats them as optional. Make the level a parameter, or log at `INFO`.

---

## 2. Design and API

### 2.1 Equality, hashing and summaries cover different sets of types (P2) **[verified]** — **Fixed (#2083)**

_Still open from earlier review item 2.1, which fixed only the scalar and standard-library types._
`hash_object(np.arange(3))` raises `TypeNotRegisteredError`, and the same happens for torch, pandas,
polars, xarray and PyArrow objects, while equality supports all of them. For the library's main use
(a stable content hash of test fixtures and experiment configs) this is the largest gap.
`hash_object` can build on the equality work: arrays hash as `(dtype.str, shape, tobytes())`,
DataFrames as their column names plus a per-column hash. Keep it opt-in per library, as in the equality
registry, so the import cost stays zero.

### 2.2 PyArrow ignores `atol`, `rtol` and `equal_nan` (P2) **[verified]** — **Fixed (#2081)**

`objects_are_allclose(pa.table(...), pa.table(...))` warns and compares exactly, and with
`equal_nan=True` two arrays of `nan` are unequal (`False`). This is documented and
warned about, so it is not a bug, but it is the only major backend with that limit. Converting the
numeric columns with `to_numpy(zero_copy_only=False)` and delegating to the NumPy tester (the pandas tester
probably already has the right structure) would close it. If not, make the warning an opt-out
error (`config.strict=True`) so that a test cannot pass on an exact comparison when the user asked for a
tolerance.

### 2.3 Python 3.15 is in the classifiers while the version is `1.2.0a2` (P3) — **Partly fixed (#2080)**

`pyproject.toml` declares `Development Status :: 5 - Production/Stable` for an alpha version, and lists 3.15.
Either drop the 3.15 classifier until it is in the required CI matrix, or confirm the matrix has it.
The `description` ("Library to check equality between two complex/nested objects") undersells the package:
it now ships hashing, summaries, identifiers, I/O, random-state management and filesystem utilities. The
keywords are all about array libraries; add `hashing`, `summary`, `identifiers`.

### 2.4 Scope: `utils` keeps growing into unrelated territory (P3) — **Open**

_Related to scope item 1 of `src-coola-review.md`._ `utils/git.py`, `utils/password.py`, `utils/secret.py`,
`utils/bloom_filter.py`, `utils/file_size.py`, `utils/env_vars.py` and `identifier/` have nothing to do
with comparing nested objects and are exercised by none of the core modules. They add API surface to
version, document and secure (item 1.1 is a leak in one of them). Options: move them into a clearly
labelled `coola.contrib` or a separate `coola-utils` distribution, or mark them in the docs as
"not part of the stable API" until that is decided.

### 2.5 `random` has no way to scope a seed across all backends in one context manager (P3) — **Open**

`random_seed`, `numpy_seed` and `torch_seed` are per-backend; `manual_seed` is global and permanent. The
common test need is "seed everything and restore afterwards". A single `seed_everything(seed)` context
manager built on `get_rng_state` / `set_rng_state`, whose default is the registry's managers, would remove
code that every user writes by hand.

---

## 3. Tests, tooling and CI

- **Add regression tests for every [verified] item above.** Each is a one- or two-line repro and
  they are listed in this file.
- **Property test for the cross-module invariants.** Not yet present, and cheap with Hypothesis:
  `hash_object(x) == hash_object(deepcopy(x))`, `objects_are_equal(a, b)` implies
  `hash_object(a) == hash_object(b)` for hashable types, `summarize(x)` never raises for any generated
  value, and `convert_to_jsonable(x)` is accepted by `json.dumps(..., allow_nan=False)`. The last two would
  have caught 1.2 and 1.5.
- **Windows and `newline` behaviour.** The classifiers list only POSIX and macOS, and 1.6 only shows
  on Windows in the other direction. Either add a Windows runner for the `io` tests or state in the README
  that Windows is untested.
- **Generated files in the working tree.** `coverage.xml` and `htmlcov/` are ignored by git, so this is
  only a note: `codespell` skips them explicitly in `pyproject.toml`, which suggests they get in the way
  of local runs. `make clean` (or an `invoke` task) that removes them would be convenient.
- _Still open from the earlier reviews:_ one mutation-testing run (best on `equality/handler`), a coverage
  `fail_under`, a second type checker in CI, and `pytest-xdist` if the matrix grows.

---

## 4. Suggested order for Part A

1. 1.1 (secret leak) and 1.2 (`summarize` crash): one-line fixes with a clear test each.
2. 1.3, 1.4 (equality returns or crashes), 1.7 (partial seeding).
3. 1.5 and 1.6 (JSON and text round trips): decide the contract first, then fix.
4. 2.1 (hashing for array and table types) and 2.2 (PyArrow tolerance): the two largest feature gaps.
5. 2.3 to 2.5 and the small leads in 1.8 to 1.10, as cleanup.

---

# Part B: full read-through findings

Item IDs are `<section>.<n>`. Paths are relative to `src/coola/` unless noted.

## 5. `equality`

| ID   | P   | Finding                                                                                                                                                                                                                                                                                                                                                                                           |
| ---- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1  | P1  | **[verified]** Object-dtype ndarray holding arrays raises `ValueError` (`handler/numpy.py:100`, `np.array_equal`). Compare element-wise through the registry for `dtype.kind == "O"` (also fixes 1.3).                                                                                                                                                                                            |
| 5.2  | P2  | **[verified]** `==` that does not return a bool crashes: `pd.Index`, `MultiIndex`, `Categorical`, extension arrays, scipy sparse give `TypeError` for copies, but `True` for the same object. Register `pd.Index` (`.equals` + dtype + names) and `Categorical`/`ExtensionArray`, or add an `.equals` fallback.                                                                                   |
| 5.3  | P2  | **[verified]** `tester/registry.py:149-156`: a dataclass with its own `__eq__` is still compared field by field (`__dataclass_params__.eq` is True when the user defines `__eq__`). The comment says the custom `__eq__` is respected. Check that `__eq__` was generated.                                                                                                                         |
| 5.4  | P2  | **[verified]** `handler/mapping.py:51-53`: keys that are `==` but of different type are the same key (`{1: "a"}` equals `{True: "a"}` and `{1.0: "a"}`), unlike sequences (`[1]` vs `[True]` is `False`). NaN keys never match, even with `equal_nan=True`. Compare keys as `(type(k), k)`.                                                                                                       |
| 5.5  | P2  | **[verified]** Polars: `equal_nan=False` is ignored for NaN inside `List`/`Array`/`Struct` columns (`handler/polars.py:168-185`); flat float columns are handled. Detect NaN recursively or document.                                                                                                                                                                                             |
| 5.6  | P2  | **[verified]** `SameObjectHandler` (`handler/native.py:270`) makes NaN comparison depend on identity: `[n]` equals `[n]` but not `[float("nan")]`; an array with NaN equals itself, not its copy. Document it on `objects_are_equal`, or skip the shortcut when `equal_nan=False` for float types.                                                                                                |
| 5.7  | P2  | **[verified]** Scalars: `objects_are_allclose(1, 1.0)` and `([1.0], [1])` are `False`; `objects_are_allclose(True, False, atol=2)` is `True` while `np.bool_` is deliberately not tolerant. Register `bool` with the default tester and let real numbers compare across int/float (or document).                                                                                                  |
| 5.8  | P2  | **[verified]** `compare()` paths ignore dataclass fields and attributes (`handler/dataclass.py:67-77`, `native.py:195-205`, `data.py:60-68`): the path of a difference inside `E(1, [1])` is `(0,)`, not `("y", 0)`, so `format_path` prints a wrong location. `extra={"coola_path_kind": ...}` is written (`sequence.py:102`, `mapping.py:172`) and never read: use it or delete it.             |
| 5.9  | P2  | **[verified]** `handler/format.py:67`: `if different_value_key is not None` yields an empty message when the differing key is `None`. Use a sentinel.                                                                                                                                                                                                                                             |
| 5.10 | P2  | **[verified]** `handler/base.py:299-317` rejects only direct cycles (`handler is self`). `a -> b -> a` is accepted, then `get_chain_length`/`visualize_chain` loop forever. `validate_chain()` exists but nothing calls it; call it in `set_next_handler` and `create_chain`.                                                                                                                     |
| 5.11 | P2  | **[verified]** `handler/pyarrow.py:96-99`: a plain `objects_are_allclose` warns that `atol`/`rtol` are ignored because the defaults are non-zero. Warn only if the arrays differ, and see 2.2. `stacklevel=4` points inside coola.                                                                                                                                                                |
| 5.12 | P2  | **[verified]** `result.py:~148`: `compare()` calls `repr()` on both objects (a raising `__repr__` crashes `assert_objects_equal`, and a huge list is rendered twice for a message that never shows it). Use `summarize` with a fallback, or build it lazily.                                                                                                                                      |
| 5.13 | P3  | **[verified]** `max_depth=1000` can never trigger: about 6 frames per level means a depth of 200 already raises `RecursionError`. Lower the default or derive it from `sys.getrecursionlimit()`.                                                                                                                                                                                                  |
| 5.14 | P3  | **[verified]** `handler/scalar.py:125`: `objects_are_allclose(2**60, 2**60 + 1, rtol=0, atol=1e-8)` is `True` (ints converted to `float`); use exact integer arithmetic when both are `int`. With `equal_nan=True`, `complex(nan, 1)` equals `complex(nan, 2)`; compare the parts separately.                                                                                                     |
| 5.15 | P3  | Cleanups: `rtol`/`atol` validated twice (`interface.py`, `config.py`); interleaved comments in `config.py:84-99`; `type(seq) in {list, tuple}` builds a set per call (`handler/sequence.py:111`); the fast path skips lists of floats (73 ms vs 11 ms for 200k items); chains with no terminal handler fail only at compare time; `handler/polars.py:313-322` is probably dead for `polars>=1.0`. |

## 6. `hashing` and `identifier`

| ID   | P   | Finding                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---- | --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.1  | P1  | **[verified]** `hash_object(10**5000)` and `Fraction(10**5000, 3)` raise the int-to-str limit `ValueError` (`hashing/repr.py`, `canonical.py`). Hash `format(n, "x")` (not limited). Same family as 1.2.                                                                                                                                                                                                   |
| 6.2  | P1  | **[verified]** `DecimalHasher` (`canonical.py:49`) uses `normalize()`, which rounds to the context precision (28 digits): two different decimals collide, and the result depends on the thread-local context. Normalise from `as_tuple()`.                                                                                                                                                                 |
| 6.3  | P1  | **[verified]** `DatetimeHasher` (`datetime.py:80`) raises `OverflowError` for aware datetimes near `min`/`max` (`astimezone(UTC)`). Compute `naive - utcoffset()` or fall back to ISO with offset.                                                                                                                                                                                                         |
| 6.4  | P1  | **[verified]** `SnowflakeIdGenerator` can mint a duplicate and a non-monotonic ID when the sequence wraps: the lock is released during the sleep without committing a timestamp, so a second thread mints `(T, 1)` again (`identifier/snowflake.py:~165-205`). Sleep under the lock or commit `timestamp + 1`.                                                                                             |
| 6.5  | P2  | **[verified]** `@cache` on `get_default_generator` (`snowflake.py`, `objectid.py`) does not serialise the first call: two threads build two generators with the same `worker_id`. Use `LazySingleton`.                                                                                                                                                                                                     |
| 6.6  | P2  | **[verified]** `generate_snowflake_id(timestamp_ms=future)` stores the value as `_last_timestamp_ms` in the shared default generator; later calls raise "clock moved backward" until restart. Do not update state for an explicit timestamp (as `generate_ulid` does).                                                                                                                                     |
| 6.7  | P2  | **[verified]** `MappingHasher` (`mapping.py:99-100`) sorts by key hash only; keys with equal hashes (unhashable keys with `ignore_unhashable=True`, short `length`) keep insertion order, so equal dicts hash differently. Sort by the full `key_hash + val_hash`.                                                                                                                                         |
| 6.8  | P2  | **[verified]** Enum mixins of `bytes`, `datetime`, `tuple`, `Mapping` hash as the plain base value (only `str`/`int`/`float` check `Enum`). Route every `Enum` to `EnumHasher` first in `HasherRegistry.hash`.                                                                                                                                                                                             |
| 6.9  | P2  | **[verified]** Any object with a callable `hash` attribute is taken as `SupportsHash` (`interface.py:_has_hash_method`): `hash_object(pl.Series([1]))` raises `TypeError` that `ignore_unhashable` does not catch. Check the signature, or require a marker.                                                                                                                                               |
| 6.10 | P2  | Hashes include `__module__`/`__qualname__` (`dataclass.py:85`, `pydantic.py`, `hashable.py`, `string.py`): a class defined in `__main__` hashes differently from the imported one, and renaming a class changes persisted hashes and `generate_stable_uuid5` IDs. Document, or allow a stable tag.                                                                                                         |
| 6.11 | P2  | `ReprHasher` (`repr.py:125`) tags by bare `__qualname__`, so a user class named `Decimal` or `Path` shares the tag namespace with the built-in hashers. Use `module.qualname`.                                                                                                                                                                                                                             |
| 6.12 | P2  | `generate_stable_content_id` and `generate_stable_uuid5` still take positional `registry, length, ignore_unhashable`, while `hash_object` made them keyword-only; `generate_stable_content_id(data, 32)` fails with `AttributeError`. Make them keyword-only.                                                                                                                                              |
| 6.13 | P2  | Nested pydantic models bypass the cycle guard (`pydantic.py`, `hash_nested`): a self-referencing model gives `RecursionError` instead of the documented `ValueError`, and a model inside a dataclass ignores the outer `on_secret` policy. Route through `registry.hash`.                                                                                                                                  |
| 6.14 | P2  | **[verified]** Deep nesting (2000 levels) raises a raw `RecursionError` in `hashing/registry.py`.                                                                                                                                                                                                                                                                                                          |
| 6.15 | P2  | **[verified]** `extract_ulid_timestamp_ms("Z" * 26)` returns a 50-bit "timestamp" (the first character must be `<= 7`), and lowercase ULIDs are rejected although Crockford decoding is case-insensitive. Validate `< 2**128`; normalise in `decode_crockford_base32`.                                                                                                                                     |
| 6.16 | P3  | **[verified]** `verify_checksummed_id` accepts non-ASCII look-alikes (`ı` upper-cases to `I`, then maps to `1`) and checks `sep` case-sensitively. Reject non-ASCII input first.                                                                                                                                                                                                                           |
| 6.17 | P3  | No upper bound on `generate_checksummed_id(length=...)` or on `decode_obfuscated_id` input (quadratic big-int decode; a valid value has at most 11 base62 characters). `nanoid` already caps at 1024.                                                                                                                                                                                                      |
| 6.18 | P3  | Obfuscated IDs: bit 0 of the output is `(n + c) & 1`, so consecutive IDs alternate parity. The docs claim to hide ordering and magnitude; add the caveat or mix with a Feistel step.                                                                                                                                                                                                                       |
| 6.19 | P3  | `hash_bytes`/`hash_string` duplicate the length validation; validate `length` before hashing a whole structure; `hash_bytes` rejects `memoryview`.                                                                                                                                                                                                                                                         |
| 6.20 | P3  | `get_default_generator` is in `__all__` of both `objectid.py` and `snowflake.py` (different objects): star imports clobber each other.                                                                                                                                                                                                                                                                     |
| 6.21 | P3  | Minor: `chars` declared twice in `_encode_base62`; `get_default_hasher_registry` and `get_default_registry` both exported; `generate_prefixed_id` accepts spaces, `/` and non-ASCII in the prefix and a non-`str` generator result; the `register_hashers` docstring example pollutes the global registry under doctest; `PureWindowsPath("a/b")` and `PurePosixPath("a/b")` hash the same **[verified]**. |

## 7. `io` and `utils`

| ID   | P   | Finding                                                                                                                                                                                                                                                                                                                                                    |
| ---- | --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.1  | P1  | **[verified]** `io/base.py:107,130-138`: the temporary file name (`.{name}-{32 hex}{ext}`) and lock name (`{name}.lock`) are longer than the target, so saving to a legal 221+ character file name raises `OSError: File name too long`. Use a short fixed prefix and a hash for the lock name.                                                            |
| 7.2  | P1  | **[verified]** `utils/text_diff.py:58-64`: `unified_diff("a", "b")` returns `...-a+b`: texts without a trailing newline get their last lines glued together. Add the missing `\n` (git prints `\ No newline at end of file`).                                                                                                                              |
| 7.3  | P2  | `io/base.py:47-51,107-125`: the lock acquire timeout (60 s) is shorter than the stale-lock threshold (300 s), so a crashed writer makes every save to that path fail for 4 minutes. The timeout is not configurable. Record the pid and check it, or use `fcntl`/`msvcrt` locks.                                                                           |
| 7.4  | P2  | **[verified]** `io/base.py:482-483`: `exist_ok=True` resets the mode of an existing file (0600 becomes 0644) and replaces a symlink with a regular file. Copy the mode and resolve the symlink.                                                                                                                                                            |
| 7.5  | P2  | Stale-lock breaking has a check-then-rename race that can steal a live lock (`io/base.py:75-82`, `165-178`). Not reproduced. Re-check the renamed file or use kernel locks.                                                                                                                                                                                |
| 7.6  | P2  | **[verified]** `utils/git.py:145-148`: `is_git_repo(".git")` is `True`; `--is-inside-work-tree` prints `false` there and the output is ignored. Compare to `"true"`. `run_git_command` also drops git's stderr from the error (`git.py:69-71`).                                                                                                            |
| 7.7  | P2  | **[verified]** `to_date(datetime)` returns the datetime unchanged (`utils/date.py:90-92`), so it does not equal `date(...)`. Call `.date()`.                                                                                                                                                                                                               |
| 7.8  | P2  | **[verified]** `slugify("日本語")` is `''` and `slugify("Straße")` is `'strae'` (`utils/string.py:121-126`). Fall back to a hash when empty; map `ß` before the ASCII encode.                                                                                                                                                                              |
| 7.9  | P2  | **[verified]** `to_jsonable` returns `datetime.time` and `timedelta` unchanged (`utils/conversion.py:104-110`). Add them.                                                                                                                                                                                                                                  |
| 7.10 | P2  | **[verified]** `check_env_vars` treats `X=""` as defined; `get_required_env_var` treats it as missing (`utils/env_vars.py:46-47` vs `108`).                                                                                                                                                                                                                |
| 7.11 | P2  | **[verified]** `repr_mapping(..., sorted_keys=True)` and `sort_by_keys` raise `TypeError` for mixed key types (`utils/format.py:467-469`, `utils/mapping.py:26`). Sort by `(type name, key)`.                                                                                                                                                              |
| 7.12 | P2  | `to_tensor` without torch raises `AttributeError: module 'torch' has no attribute 'is_tensor'` instead of `MissingPackageError`: the fake torch lacks `is_tensor`, `as_tensor`, `from_numpy`, `ones` (`utils/fallback/torch.py:26-34`). Call `check_torch()` first.                                                                                        |
| 7.13 | P2  | **[verified]** `LazyModule` (`utils/imports/lazy.py:42-45`): `hasattr(LazyModule("missing"), "a")` raises `ModuleNotFoundError`; `copy.copy` fails. Raise `AttributeError` for dunder names and guard `_module`.                                                                                                                                           |
| 7.14 | P3  | **[verified]** `LazySingleton` rebuilds on every call if the factory returns `None` (`singleton.py:64-84`); use a sentinel.                                                                                                                                                                                                                                |
| 7.15 | P3  | `CaptureHandler` only records from the creating thread and drops records below INFO (`utils/logging.py:35-41`).                                                                                                                                                                                                                                            |
| 7.16 | P3  | **[verified]** `sanitize_path` silently drops the host of `file://server/share/x` and raises `RuntimeError` for an unknown `~user` (`utils/path.py:112-114`). `find_root_package_parent` returns a non-existent file path itself (`path.py:55-71`).                                                                                                        |
| 7.17 | P3  | **[verified]** `str_human_byte_size(1048575)` gives `'1,024.00 KB'` (pick the unit after rounding) (`utils/format.py:424-425`).                                                                                                                                                                                                                            |
| 7.18 | P3  | **[verified]** `TorchLoader() != TorchLoader(weights_only=True)` although identical in behaviour: the default is applied in `load`, not `__init__` (`io/torch.py:62-71`).                                                                                                                                                                                  |
| 7.19 | P3  | **[verified]** `save_text("x", d/"file"/"g.txt")` where `file` is a regular file raises `FileExistsError`, which reads like `exist_ok=False` (`io/base.py:465`). Raise `NotADirectoryError`.                                                                                                                                                               |
| 7.20 | P3  | `_fsync_file` opens the file `O_RDONLY`; `fsync=True` fails on Windows (`io/base.py:141-147`).                                                                                                                                                                                                                                                             |
| 7.21 | P3  | `check_torch_numpy` raises `MissingPackageError` without `name` and with a dangling space (`utils/imports/torch_numpy.py:44-49,90-93`). `package_available` suppresses only `ModuleNotFoundError` (`universal.py:99-100`); also catch `ValueError`/`ImportError`.                                                                                          |
| 7.22 | P3  | `get_package_version` raises on non-PEP 440 strings and its `lru_cache` is not cleared by `clear_availability_caches` (`utils/version.py:48-53`).                                                                                                                                                                                                          |
| 7.23 | P3  | Wrong docstring examples that fail as doctests: `repr_sequence` (`format.py:155`), `universal.py:204-205`, `env_vars.py:99-117`.                                                                                                                                                                                                                           |
| 7.24 | P3  | Minor: `float()` of ints above 2**53 in `stats.py:60`; `to_array`/`to_tensor` share memory for arrays and copy for lists (undocumented); `to_array(bfloat16 tensor)` raises; `working_directory` is process-wide (no thread warning); `BaseFileSaver.save` has `fsync` which `BaseSaver.save` lacks; the in-process lock key is `str(path)`, not resolved. |

## 8. `registry`, `summary`, `factory`, `validation`, `display`, `testing`

| ID   | P   | Finding                                                                                                                                                                                                                                                                                                                    |
| ---- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8.1  | P1  | **[verified]** `summarize` of a polars `LazyFrame` with an invalid plan raises `ColumnNotFoundError`, and `collect_schema()` resolves the whole plan (file metadata of `scan_parquet`/`scan_csv`, possibly remote) (`summary/polars.py:~96`). Wrap in `try/except` and say so in the docstring.                            |
| 8.2  | P1  | **[verified]** `TypeRegistry.register("abc", ...)` accepts a non-type key; every later miss then raises `AttributeError` instead of `TypeNotRegisteredError` (`registry/type.py:152-160,297`). Validate keys in `register`/`register_many`/`initial_state`.                                                                |
| 8.3  | P2  | **[verified]** `str_pydantic_model(m, exclude_fields=["token"])` warns that an existing `SecretStr` field "does not exist" (the check runs after secrets/`None` were removed); the docstrings say unknown names are "silently ignored" (`display/pydantic.py:96-104`).                                                     |
| 8.4  | P2  | **[verified]** `exclude_secret` handles `SecretStr` only: any generic containing it (`tuple[SecretStr, int]`) drops the whole field, but `SecretBytes` and `Secret` are shown masked (`display/pydantic.py:53-57`).                                                                                                        |
| 8.5  | P2  | **[verified]** `configure_colorlog_logging` differs on a TTY: `stream=`, `handlers=` fail or are ignored, `format=` is silently ignored; `sys.stderr = None` raises `AttributeError` (`display/colorlog.py:61-87`).                                                                                                        |
| 8.6  | P2  | **[verified]** `factory`: an `allowed_prefixes` violation, or any `ImportError` inside the target module, is reported as "target object does not exist"; a `str` `allowed_prefixes` is iterated per character (`factory/resolve.py:~83-88`).                                                                               |
| 8.7  | P2  | **[verified]** `import_object("")` raises `ValueError`, `resolve_object(1, (int, str))` raises `AttributeError` (tuple has no `__qualname__`), `is_object_config("x", Foo)` raises `AttributeError`; a `SyntaxError` in a string annotation escapes (`factory/instantiation.py`, `resolve.py`, `config.py`).               |
| 8.8  | P2  | **[verified]** `is_object_config` rejects functions annotated `-> list[int]` (`inspect.isclass` is `False` for generic aliases) (`factory/config.py:~82-85`). Use `get_origin(t) or t`.                                                                                                                                    |
| 8.9  | P2  | **[verified]** The depth-limited preview changes the type: a tuple or set longer than the preview shows as a list, a `Counter` as a `dict`; `max_items < 0` still previews 5 items (`summary/collection.py:84-93`, `sequence.py`, `set.py`, `mapping.py`).                                                                 |
| 8.10 | P2  | **[verified]** `BaseTypeDispatchRegistry` (so `SummarizerRegistry` and 5 other registries) has no `unregister`, `clear`, `__len__`, `__contains__`, `items` or `equal`; `register_summarizers` on the global is irreversible without touching `_state` (`registry/dispatch.py`).                                           |
| 8.11 | P2  | **[verified]** Validators break their documented contract: `validate_finite(10**400)` raises `OverflowError`, `validate_finite(None)` and `validate_non_negative("a")` raise `TypeError`, `validate_non_negative(True)` passes, an ndarray gives "truth value is ambiguous" (`validation/comparison.py`, `numeric.py:28`). |
| 8.12 | P3  | **[verified]** Not-registered message lists a class with the same `__qualname__` as the missing one; duplicates message order comes from a `set` (`registry/type.py:157-172`, `base.py:128`).                                                                                                                              |
| 8.13 | P3  | **[verified]** `TypeRegistry._cache` holds strong references to resolved dynamic classes (bounded at 1024) (`registry/type.py:263-265`).                                                                                                                                                                                   |
| 8.14 | P3  | `KeyNotRegisteredError(KeyError, LookupError)` has a redundant base. Stale docstrings in `registry/dispatch.py` (`register_many`, `KeyError` vs `TypeNotRegisteredError`, "default handler").                                                                                                                              |
| 8.15 | P3  | Export gaps: `summary/__init__.py` omits `BaseMetadataSummarizer`, `BaseFieldsSummarizer`, `format_fields`, `MAX_FIELDS`; `display/__init__.py` omits `secret_field_names` and `configure_colorlog_logging` (a logging helper in a "display" package); the `Summarizer` protocol has no `equal`.                           |
| 8.16 | P3  | Dead registrations in `summary/interface.py:~160-170` (`str`, `int`, `float`, `complex`, `bool` map to the `object` summarizer); `_get_jax_summarizers` allocates a device array, which initialises the JAX backend as a side effect.                                                                                      |
| 8.17 | P3  | **[verified]** `import coola.summary` takes about 1.7 s and imports every installed array library; `import coola.registry` imports pydantic through `coola.display`. Register lazily by qualified type name.                                                                                                               |
| 8.18 | P3  | `summarize(max_depth<0)` is accepted although the docstring says "Must be non-negative" (`summary/interface.py`).                                                                                                                                                                                                          |
| 8.19 | P3  | `BaseRegistry.equal` runs the user comparison while holding both locks (`registry/base.py:187-190`), which can deadlock with another thread comparing the reverse pair. Snapshot, release, then compare.                                                                                                                   |
| 8.20 | P3  | `testing/fixtures.py`: no defect. Names served by `__getattr__` are not in `dir(module)`; add `__dir__` if tooling matters.                                                                                                                                                                                                |

## 9. `iterator`, `recursive`, `nested`, `reducer`, `random`

| ID   | P   | Finding                                                                                                                                                                                                                                                                                                                                                                                   |
| ---- | --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8.10 | PR open (#2191)                 | `BaseTypeDispatchRegistry` has `unregister`, `clear`, `items`, `equal`, `__len__`, `__contains__`.                                                                                                    |
| 8.11 | Mostly (#2192)                  | `validate_finite` accepts huge ints; uncomparable values raise one clear `TypeError`. `validate_non_negative(True)` still passes (deliberate).                                                        |
| 8.12 | PR open (#2193)                 | Same-name classes qualified with their module; duplicates sorted.                                                                                                                                     |
| 8.13 | PR open (#2199)                 | The resolution cache holds weak references.                                                                                                                                                           |
| 8.14 | PR open (#2194)                 | Redundant `LookupError` base removed; dispatch docstrings fixed.                                                                                                                                      |
| 8.15 | Mostly (#2197)                  | Missing names of `coola.summary` exported. `coola.display` left unchanged (by decision); the `Summarizer` protocol still has no `equal`.                                                              |
| 8.16 | Mostly (#2198)                  | No JAX array created at registration. Dead scalar registrations kept (documented).                                                                                                                    |
| 8.17 | Open                            | Not done: needs lazy registration by qualified type name and lazy imports in every summarizer module.                                                                                                 |
| 8.18 | PR open (#2196)                 | `summarize(max_depth<0)` raises `ValueError`.                                                                                                                                                         |
| 8.19 | PR open (#2195)                 | `BaseRegistry.equal` compares snapshots without holding the locks.                                                                                                                                    |
| 9.1  | P1  | **[verified]** `from_flat_dict({"a": d, "a.y": 2})` mutates the caller's dict `d`; with `restore_lists=True` a user-supplied dict leaf can be turned into a list (`nested/flat.py`, `_set_nested`). Track the dicts the function created and raise "Key conflict" for others.                                                                                                             |
| 9.2  | P1  | **[verified]** `UserString` crashes every traversal: `dfs_iterate` and `recursive_apply` raise `RecursionError`, `bfs_iterate` a false "Cycle detected". It is a real `Sequence` of `UserString`s. Register it as a leaf in `iterator/bootstrap.py` and `recursive/interface.py`.                                                                                                         |
| 9.3  | P2  | **[verified]** `flatten_mapping(always_prefix=True)` and `"prefix"` overwrite colliding keys silently (`nested/mapping.py`). Raise `DuplicateKeyError`.                                                                                                                                                                                                                                   |
| 9.4  | P2  | **[verified]** `expand_list_columns` replaces an existing column named `l.0`, and drops the column when every list is null or empty (`nested/polars.py:~120`).                                                                                                                                                                                                                            |
| 9.5  | P2  | **[verified]** `flatten_frame` with two list-of-struct columns raises `ShapeError` (parallel `explode`); `depth=0` is rejected although it is a sensible no-op (`nested/polars.py`, `unnest_one_level`).                                                                                                                                                                                  |
| 9.6  | P2  | **[verified on this machine]** `TorchRandomManager` does not save or restore the MPS generator, so `torch_seed()` leaves it changed; `set_rng_state` raises a bare `KeyError: 'torch.cuda'` on a partial dict or a changed device count (`random/torch.py`).                                                                                                                              |
| 9.7  | P2  | **[verified]** The BFS cycle check is quadratic in depth (`iterator/bfs/registry.py`, `_is_ancestor`): depth 20000 takes 4.2 s, 40000 takes 18 s. DFS and `recursive_apply` raise `RecursionError` at about 1000 levels; document it.                                                                                                                                                     |
| 9.8  | P2  | **[verified]** Resolution is by MRO only, so virtual `Mapping`/`Collection` members are leaves: `MappingProxyType`, `dict.values()`, `range` (in `recursive_apply`). `recursive_apply(UserList([1]), f)` returned a plain `list`. Register `MappingProxyType` and document the rule.                                                                                                      |
| 9.9  | P3  | **[verified]** `merge_mappings(on_duplicate="suffix")` compares only with the first value (`[{"a":1},{"a":2},{"a":2}]` stores `a_2` again), NaN values never merge, and a genuine `a_1` raises under `"first"`/`"last"`. Add `equal_nan`. `flatten_mapping(on_duplicate="prefix")` is order dependent.                                                                                    |
| 9.10 | P3  | **[verified]** `add_prefix_suffix_to_keys(recursive=True)` ignores lists of dicts, while `remove_keys_if` descends into sequences.                                                                                                                                                                                                                                                        |
| 9.11 | P3  | **[verified]** `to_flat_dict` flattens a `range` but keeps a `set`, and raises `RecursionError` for cycles where the iterators raise `ValueError`. `from_flat_dict` does not restore a top-level list.                                                                                                                                                                                    |
| 9.12 | P3  | **[verified]** `convert_to_list_of_dicts({"a": "ab", "b": [1, 2]})` splits the string per character; `[{}, {}]` does not round trip.                                                                                                                                                                                                                                                      |
| 9.13 | P3  | **[verified]** `NumpyReducer.std(np.ones((1, 3)))` is `nan` (`len(values) <= 1`; use `np.size`).                                                                                                                                                                                                                                                                                          |
| 9.14 | P3  | **[verified]** Reducers disagree: `median([1, 2, 3])` is `2` (Native, Torch) and `2.0` (Numpy); `max([2**63, 1])` is an int, a lossy float, or an overflow error; `median([1e308, 1e308])` is `inf` everywhere; generators fail with three different exceptions. `auto_reducer()` returns the torch reducer for plain lists whenever torch is importable (slower, and fails above 2**63). |
| 9.15 | P3  | `_SCALAR_TYPES` in `iterator/bootstrap.py` and `recursive/interface.py`: with MRO resolution everything except `object` is dead (keep `str`/`bytes` only if documented). `DefaultChildFinder` and `DefaultIterator` express "leaf" differently, so a custom child finder with no children silently drops the element in BFS.                                                              |
| 9.16 | P3  | `random_seed(..., manager=...)` uses `manager or get_default_registry()`; use `is None`.                                                                                                                                                                                                                                                                                                  |
| 9.17 | P3  | **[verified]** `KeyFilterTransformer(transform_func=...)`: without a predicate, `func` is reused as the key predicate and leaves become booleans. Split into two classes or drop the mode (see 2.6 of the previous review).                                                                                                                                                               |

## 10. Tests, docs, CI and packaging

| ID    | P   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----- | --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10.1  | P1  | **[verified]** `docs/docs/uguide/summary.md:498-515` fails its own doctest: the registry repr omits `collections.deque -> SequenceSummarizer`, and depends on test order (earlier examples register `Person`). Print only the relevant keys, or use a fresh registry and unregister (needs 8.10). `ci-doctest.yaml` should be red on this.                                                                                                                                                                              |
| 10.2  | P2  | `xfail` cases in `tests/integration/equality/test_external.py:48-80` run the numpy/torch assertion first, so `objects_are_equal(...)` never runs for the 3 xfail ids; they are non-strict. Split the tests and use `xfail(strict=True, raises=...)`.                                                                                                                                                                                                                                                                    |
| 10.3  | P2  | Unseeded `np.random` data in `tests/unit/utils/test_stats.py:50-56`, generated at collection time: failures cannot be reproduced. Use `default_rng(0)`.                                                                                                                                                                                                                                                                                                                                                                 |
| 10.4  | P2  | pytest configuration is minimal: add `--strict-markers`, `--strict-config`, `xfail_strict = true`, `filterwarnings = ["error", ...]` and a default `timeout` (only `inv test.*` passes one; `ci-test-properties.yaml` has none).                                                                                                                                                                                                                                                                                        |
| 10.5  | P2  | Benchmarks run as ordinary tests (7 tests, about 6 s) and the nightly job never saves or compares results (no `--benchmark-save`/`--benchmark-compare`), so it cannot catch a regression. Only equality and registry are benchmarked.                                                                                                                                                                                                                                                                                   |
| 10.6  | P2  | The YAML, Markdown, TOML and Makefile lint/format targets exist in the `Makefile` but are not run in `ci-format.yaml` or pre-commit.                                                                                                                                                                                                                                                                                                                                                                                    |
| 10.7  | P2  | The release workflows are not gated on tests: a `v*.*.*` tag publishes immediately, `workflow_dispatch` from `main` also publishes to PyPI without a tag, and `release-docs.yaml` publishes `stable` docs from any ref. Call `ci.yaml` with `workflow_call` and add `needs:`.                                                                                                                                                                                                                                           |
| 10.8  | P3  | **[verified]** 28 PyArrow tolerance tests fail under `-W error`: they use `warnings.catch_warnings(record=True)` without `simplefilter("always")` (`tests/unit/equality/tester/test_pyarrow.py:497-508`, `handler/test_pyarrow.py`). Use `pytest.warns`.                                                                                                                                                                                                                                                                |
| 10.9  | P3  | `inv test.doctest` globs every `*.md`, including `review/*.md` and the synced `.make/` tree. The `identifier.md` example triggers a `UserWarning` (run the markdown doctests with `-W error::UserWarning`).                                                                                                                                                                                                                                                                                                             |
| 10.10 | P3  | Stale tooling docs: `setup-doc-deploy` and `verify-installed-package` actions name workflows that do not exist; `ci.yaml`, `nightly-package.yaml` and `ci-test-deps.yaml` cite a missing `dev/config/test_matrix.json`; the benchmark file says `inv benchmark` (the task is `inv test.benchmark`); `ci-verify-workflows.yaml` says it triggers on workflow changes but has no `paths:` filter; `tests/TESTING_GUIDE.md` and `CONTRIBUTING.md` omit `properties/`, `benchmarks/`, `--timeout`, doctests and CUDA tests. |
| 10.11 | P3  | `examples/` has one script (`equality.py`) and its test only checks the exit code.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 10.12 | P3  | **[verified]** `tests/unit/properties/` holds only an `__init__.py`; 9 source modules have no mirrored test file (`summary/torch.py` is tested in `test_tensor.py`); two copies of the equality test helpers and the BFS/DFS samples.                                                                                                                                                                                                                                                                                   |
| 10.13 | P3  | Property tests cover equality, hashing, identifier, nested, reducer, registry and validation but not `summary`, `io`, `random`, `iterator`, `recursive`, `utils`; they run without `--cov` and with the default `max_examples`. Add a CI profile with `derandomize=True` and `print_blob=True`.                                                                                                                                                                                                                         |
| 10.14 | P3  | Three CUDA tests never run (`unit/equality/handler/test_torch.py:270`, `unit/equality/tester/test_torch.py:579`, `unit/summary/test_tensor.py:209`): mock the device or use a GPU runner nightly.                                                                                                                                                                                                                                                                                                                       |
| 10.15 | P3  | Dependency matrix (lead): `ci-test-code-dep.yaml` multiplies every Python version by every sampled package version (for example `numpy 1.24.0` on Python 3.14), which have no wheels. Filter with `exclude:`. The 5-minute timeout is tight for free-threaded and macOS cells; the `min` job writes coverage nobody uploads.                                                                                                                                                                                            |
| 10.16 | P3  | `install: dev,docs` in `invoke.yaml` omits the `property` group, so a contributor following `CONTRIBUTING.md` silently skips all of `tests/properties`. Add it. `docs/site` is a build output present in the tree (check `git check-ignore`); `inv clean` should remove `htmlcov`, `coverage.xml`, `.hypothesis`.                                                                                                                                                                                                       |
| 10.17 | P3  | `bot-update-deps.yaml` opens the PR with `if: always()` without saying that tests failed. `dependabot.yml` has no `groups:`/`cooldown:` and overlaps with `bot-update-deps` and `bot-pre-commit-autoupdate`. `inv security.audit` is not run by any workflow (add a weekly `pip-audit`).                                                                                                                                                                                                                                |
| 10.18 | P3  | Docs build: `mkdocs.yml` has no `strict: true` or `validation:` block, so broken cross references do not fail CI; `mkdocs-include-markdown-plugin` is in the `docs` group but not enabled.                                                                                                                                                                                                                                                                                                                              |

---

## 11. Suggested order for Part B

1. **Crashes and data loss**: 9.1 (input mutated), 9.2 (`UserString`), 7.1 (long file names), 5.1, 6.1 to 6.3, 8.1, 8.2.
2. **Concurrency bugs in IDs and locks**: 6.4 to 6.6, 7.3, 7.5.
3. **Silent wrong results**: 5.3, 5.4, 5.8, 6.7, 6.8, 7.2, 7.4, 7.6, 7.7, 8.9.
4. **CI and test hygiene**, which stops regressions: 10.1, 10.4, 10.7, 10.8, 10.2, 10.3, 10.6.
5. **API consistency and exports** (8.5 to 8.11, 8.15, 6.12, 6.20), then the P3 items.
