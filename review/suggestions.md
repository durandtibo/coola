# Codebase review: suggestions (round 3)

This file **replaces the previous `review/suggestions.md`** (417 lines, items 1.1 to 2.10 handled).
That version is in git history (last touched by `55086b7f`). Nothing below repeats an item of that
review, `src-coola-review.md`, `noqa-review.md` or `python-311-modernization.md`, except where
an item is explicitly marked _still open_.

## How this review was done (and its limits)

I did **not** read all 27k lines of `src/coola`. After the two earlier full read-throughs, I
concentrated on running probes against the checkout (`.venv`, Python 3.14, NumPy, torch, pandas, polars,
pyarrow, xarray installed) and reading the modules those probes pointed at: `utils`, `io`,
`identifier`, `nested`, `random`, `hashing`, `summary` and the equality testers for third-party types.
Items marked **[verified]** were reproduced; the others are leads from reading. The review changed no
source file. Not covered this time: `registry/`, `factory/`, `display/`, `validation/`, the
`iterator`/`recursive` internals, the docs site, and the CI workflow contents beyond their file names.

Priority: **P1** wrong result or leak on realistic input, **P2** inconsistent or surprising behaviour,
**P3** cleanup.

---

## 1. Correctness and robustness

### 1.1 `mask_secret` leaks the secret with negative arguments **[verified]** (P1)

`utils/secret.py` never validates `show_first` / `show_last`. Python slicing turns a negative value into
"all but the last N":

```python
mask_secret("abcdefghijkl", show_first=-2)  # 'abcdefghij**********ijkl'  (10 of 12 characters shown)
mask_secret("abcdefghijkl", show_last=-2)   # 'abc***********'
```

Reject negative values with `validate_ge(..., 0)` (the `validation` module is already used elsewhere in
`utils`). Also consider whether it is acceptable that the output length reveals the secret length for
short secrets (`'*' * len(secret)`); a fixed-width mask is the safer default for a function named
`mask_secret`.

### 1.2 `summarize` raises on values that cannot be converted to `str` **[verified]** (P1)

`summarize` is the function used to build error and log messages, so it should never be the thing that
fails.

```python
summarize(10**5000)                                   # ValueError: Exceeds the limit (4300 digits) ...
summarize(type("B", (), {"__repr__": lambda s: 1 / 0})())  # ZeroDivisionError
```

The default summarizer calls `str(data)` / `repr(data)` before truncating. Wrap that call in a
`try/except Exception` that falls back to `<ClassName object at 0x...>`, and truncate big integers by
`int.bit_length()` rather than converting them.

### 1.3 `equal_nan=True` is ignored for object-dtype NumPy arrays **[verified]** (P2)

```python
a = np.array([np.nan], dtype=object)
objects_are_equal(a, a.copy(), equal_nan=True)   # False
```

Pandas columns of `object` dtype that hold floats take the same path. Either compare object arrays
element-wise through `objects_are_equal` (so `equal_nan` and the nested testers apply), or warn like the
PyArrow tester does. Related: `np.ma.MaskedArray` with different data under the mask compares unequal
(`eq(ma([1, 2], mask=[0, 1]), ma([1, 9], mask=[0, 1]))` is `False`). That is defensible, but it is not
documented and the masked entries are not the user-visible values.

### 1.4 Torch sparse and `meta` tensors crash instead of comparing **[verified]** (P2)

```python
objects_are_equal(torch.eye(2).to_sparse(), torch.eye(2).to_sparse())   # NotImplementedError (aten::equal)
objects_are_equal(torch.empty(2, device="meta"), torch.empty(2, device="meta"))  # NotImplementedError
```

`objects_are_equal` is documented as returning a `bool`, and 1.1 of the previous review made that a
contract. Handle `layout != torch.strided` (compare `to_dense()`, or indices and values) and
`device.type == "meta"` (compare shape and dtype only), or convert `NotImplementedError` into a clear
`TypeError` that names the tensor kind. `summarize` already handles meta tensors, so the support
is inconsistent across modules.

### 1.5 `convert_to_jsonable` does not produce JSON **[verified]** (P2)

The docstring says "JSON-compatible representation", but only dataclasses and pydantic models are
converted:

```python
json.dumps(convert_to_jsonable({"a": {1, 2}}))            # TypeError (set)
json.dumps(convert_to_jsonable({"a": range(2)}))          # TypeError (range)
json.dumps(convert_to_jsonable({(1, 2): 1}))              # TypeError (tuple key)
json.dumps(convert_to_jsonable({"a": float("nan")}), allow_nan=False)  # ValueError
```

Either rename or reword (it is "dataclass/pydantic to dict"), or finish the job: `set` and `frozenset`
to sorted lists, `range` to a list, non-string keys to `str`, `nan`/`inf` to `None` or a string, `Path`
to `str`, `datetime` to ISO format. The same applies to `save_json`: int keys silently become strings on
the round trip (`{1: 'a'}` loads as `{'1': 'a'}`) and `NaN` is written as the non-standard token `NaN`.
Pass `allow_nan=False` by default or document both.

### 1.6 Text round trip is not lossless **[verified]** (P2)

`save_text("a\r\nb\rc", p); load_text(p)` returns `'a\nb\nc'`: `TextLoader` opens the file in text mode with
universal newlines, so `\r\n` and `\r` are rewritten. On Windows `TextSaver` also translates `\n` to
`\r\n`. Open both with `newline=""`, or add a `newline` option and say so in the docstring. The same
applies to `JsonLoader`/`JsonSaver`, which are less affected because JSON escapes control characters.

### 1.7 `manual_seed` can leave the generators half-seeded **[verified]** (P2)

`manual_seed(2**70)` raises `ValueError: Overflow when unpacking long long` from the torch manager
_after_ `random` and NumPy were already seeded (NumPy uses `seed % 2**32`, so it accepts anything).
Two consequences: the global state is partially changed when the call fails, and
`seed` and `seed + 2**32` give the same NumPy stream without any warning.
Validate the seed once in `manual_seed` (`0 <= seed < 2**63`, an `int`) before calling the managers, and
state the range in the docstring. Negative seeds are currently accepted by two of the three backends.

### 1.8 `str_time_human` on negative, NaN and infinite durations **[verified]** (P3)

```python
str_time_human(-5)            # '-1 day, 23:59:55'
str_time_human(float("nan"))  # ValueError: cannot convert float NaN to integer
str_time_human(float("inf"))  # OverflowError
```

The first output is the `timedelta` default and is misleading; the other two are raw conversion errors.
Handle the sign explicitly (`'-0:00:05'`) and raise a message that names the argument, like
`str_human_byte_size` and `make_bar` do. Also `str_indent("a\nb\n", 2)` returns `'a\n  b\n  '`, which
indents the empty last line and leaves trailing whitespace; skip empty lines.

### 1.9 `add_uuid_suffix` treats dots in the stem as an extension **[verified]** (P3)

`add_uuid_suffix(Path("v1.2.model"))` gives `v1-<uuid>.2.model` because `''.join(path.suffixes)` takes
every dotted part. That is deliberate for `.tar.gz`, but a version number in a file name is common.
Limit the compound suffix to a known allow-list (`.tar.gz`, `.tar.bz2`, `.tar.xz`, ...) and use
`path.suffix` otherwise.

### 1.10 Smaller correctness leads (P3)

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

### 2.1 Equality, hashing and summaries cover different sets of types (P2) **[verified]**

_Still open from earlier review item 2.1, which fixed only the scalar and standard-library types._
`hash_object(np.arange(3))` raises `TypeNotRegisteredError`, and the same happens for torch, pandas,
polars, xarray and PyArrow objects, while equality supports all of them. For the library's main use
(a stable content hash of test fixtures and experiment configs) this is the largest gap.
`hash_object` can build on the equality work: arrays hash as `(dtype.str, shape, tobytes())`,
DataFrames as their column names plus a per-column hash. Keep it opt-in per library, as in the equality
registry, so the import cost stays zero.

### 2.2 PyArrow ignores `atol`, `rtol` and `equal_nan` (P2) **[verified]**

`objects_are_allclose(pa.table(...), pa.table(...))` warns and compares exactly, and with
`equal_nan=True` two arrays of `nan` are unequal (`False`). This is documented and
warned about, so it is not a bug, but it is the only major backend with that limit. Converting the
numeric columns with `to_numpy(zero_copy_only=False)` and delegating to the NumPy tester (the pandas tester
probably already has the right structure) would close it. If not, make the warning an opt-out
error (`config.strict=True`) so that a test cannot pass on an exact comparison when the user asked for a
tolerance.

### 2.3 Python 3.15 is in the classifiers while the version is `1.2.0a2` (P3)

`pyproject.toml` declares `Development Status :: 5 - Production/Stable` for an alpha version, and lists 3.15.
Either drop the 3.15 classifier until it is in the required CI matrix, or confirm the matrix has it.
The `description` ("Library to check equality between two complex/nested objects") undersells the package:
it now ships hashing, summaries, identifiers, I/O, random-state management and filesystem utilities. The
keywords are all about array libraries; add `hashing`, `summary`, `identifiers`.

### 2.4 Scope: `utils` keeps growing into unrelated territory (P3)

_Related to scope item 1 of `src-coola-review.md`._ `utils/git.py`, `utils/password.py`, `utils/secret.py`,
`utils/bloom_filter.py`, `utils/file_size.py`, `utils/env_vars.py` and `identifier/` have nothing to do
with comparing nested objects and are exercised by none of the core modules. They add API surface to
version, document and secure (item 1.1 is a leak in one of them). Options: move them into a clearly
labelled `coola.contrib` or a separate `coola-utils` distribution, or mark them in the docs as
"not part of the stable API" until that is decided.

### 2.5 `random` has no way to scope a seed across all backends in one context manager (P3)

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

## 4. Suggested order

1. 1.1 (secret leak) and 1.2 (`summarize` crash): one-line fixes with a clear test each.
2. 1.3, 1.4 (equality returns or crashes), 1.7 (partial seeding).
3. 1.5 and 1.6 (JSON and text round trips): decide the contract first, then fix.
4. 2.1 (hashing for array and table types) and 2.2 (PyArrow tolerance): the two largest feature gaps.
5. 2.3 to 2.5 and the small leads in 1.8 to 1.10, as cleanup.
