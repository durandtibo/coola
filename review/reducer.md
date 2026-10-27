# `coola.reducer`: target state and the changes to reach it

This document describes the design that `coola.reducer` should have, then the changes to make to the
current implementation (`main`, `93071639`) to get there. The design follows the rules of the user
guide (`docs/docs/uguide/reducer.md`, **Design Principles** and **Properties**) and the architecture
guide (`docs/docs/dev/architecture.md`: lazy imports, strict contracts; `reducer` is one of the
packages that is _not_ built on `registry`), and it stays aligned with the sibling packages
(`review/recursive.md`, `review/summary.md`). Its aim is a small package that is easy to maintain:
**one base class that owns the contract, three backends that only compute**. Breaking changes are
accepted, and the future state is more important than the past one.

Evidence comes from probes run on `main` (Python 3.14, NumPy and PyTorch installed, scripts kept
outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from the
code. No prototype of the new base class was written: the sizes of the appendix are estimates.
Nothing in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 725 lines of source and 2,190 lines of tests. The
   target removes the code that exists only because the contract is spread over the backends (the
   `BaseReducer` / `BaseBasicReducer` pair with a hole for `sort`, `_is_empty` and the `_check_values`
   hook, the copies of the overflow rule of the median, two tensor converters) instead of adding code
   next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. A custom reducer written for the current hooks is ported (the
   migration page lists the change).
3. **No slower than today**, and faster where a probe shows an easy win (the mean of `NativeReducer`
   is 15 times slower than `statistics.fmean`, F8). A design that is simpler but slower is rejected.
4. **Consistent with the siblings.** Keyword-only options, strict contracts (an invalid option or
   input raises before any work, with a message that names the remedy), stateless objects that can be
   shared by threads, lazy optional libraries.
5. **Optional libraries stay lazy.** `import coola.reducer` imports neither NumPy nor PyTorch
   (`lazy_exports`, as today). `TorchReducer` costs 0.94 s of import **[verified]**, which is why
   `auto_reducer` must not choose it by default (F6).

## 1. Properties of the sub-package

A reducer turns a one-dimensional sequence of real numbers into a statistic (`max`, `min`, `mean`,
`median`, `std`, `quantile`) or into a sorted list. It exists so that code written once runs with
whatever numeric library the user has, and gives the same answer. These are the rules of the
package. Every design choice below exists to make one of them true **by construction** (in
`BaseReducer`) instead of by the discipline of each backend.

| Property                    | Statement                                                                                                                                                                                                                                                      | Enforced by                        |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Same interface              | `NativeReducer`, `NumpyReducer` and `TorchReducer` have the same methods, the same errors and the same result types; the backend can be swapped. A backend only computes.                                                                                      | `BaseReducer` (template methods)   |
| One input rule              | The input is a sequence of real numbers (`list`, `tuple`, `range`), or a one-dimensional array / tensor for the backend of the library. An iterator, a set, a string, a 0-d or n-d array is rejected with an error that says what to pass.                       | `BaseReducer`, `_prepare`          |
| Python results              | The result is a Python `int` or `float` (a `list` of them for `quantile` and `sort`), never a NumPy or torch scalar. `max`, `min` and `sort` keep the type of the elements; `mean`, `std` and `quantile` are `float`; the median of an odd number of ints is an `int`. | the backends, one contract test    |
| Explicit empty case         | `max`, `min`, `mean`, `median`, `std` and `quantile` raise `EmptySequenceError` (a `ValueError`) on an empty input. `sort([])` is `[]`. `std` of one value is `NaN`.                                                                                             | `BaseReducer`                      |
| Valid options first         | `quantiles` outside `[0, 1]` (or `NaN`) raise `ValueError` before the input is read, with the same type and message in every backend.                                                                                                                          | `BaseReducer`                      |
| NaN                         | If a value is `NaN`, `max`, `min`, `mean`, `median`, `std` and every quantile are `NaN`, whatever the position of the `NaN`. `sort` puts the `NaN` last (first if `descending`). `std` is `NaN` if a value is infinite.                                         | the backends, one contract test    |
| No silent overflow          | The median of finite values is finite (the two middle values are halved before the mean if their sum overflows).                                                                                                                                               | the backends, one contract test    |
| Unsupported values are loud | A value that the backend cannot represent (an integer beyond 64 bits for NumPy / torch, a complex number, a `Decimal`) raises `TypeError` that names the backend and, when it exists, the backend that supports it (`NativeReducer`). No leaked `AttributeError`.   | the backends' `_prepare`           |
| Pure                        | The input is never modified or consumed. A reducer is stateless, so one instance can be shared by threads and calls.                                                                                                                                           | stateless backends                 |
| Lazy backends               | `import coola.reducer` imports no optional library. `NumpyReducer()` / `TorchReducer()` raise `MissingPackageError` at construction if the library is missing.                                                                                                  | `lazy_exports`, `check_*`          |
| Automatic backend           | `auto_reducer()` returns the fastest backend that is installed for plain Python sequences: NumPy, then native. PyTorch is never chosen implicitly (it is for tensors that live on a device, which the user passes to `TorchReducer`).                            | `auto_reducer`                     |
| Extensible                  | A backend is added by subclassing `BaseReducer` and implementing `_prepare` and six computations. It cannot forget the empty, iterator, quantile or shape checks, because it does not write them.                                                              | `BaseReducer`                      |

Not guaranteed, and documented as such:

- **Bit-identical floating point** across backends. The mean of `[0.1, 0.2, 0.3]` is `0.2` for
  `NativeReducer` and `0.20000000000000004` for the other two **[verified]**. The tests compare with
  a tolerance.
- **Sums that exceed the range of a `float`.** `mean([1e308, 1e308])` is `1e308` for `NativeReducer`
  and `inf` for NumPy and torch **[verified]**.
- **Other number types** (`bool`, `Fraction`, `Decimal`): `NativeReducer` computes with what Python
  provides; NumPy and torch may reject them (`Decimal` raises `AttributeError` today **[verified]**,
  `TypeError` in the target).
- **Speed of PyTorch** for a Python list: the values are converted to a tensor first (110 ms for the
  maximum of a list of one million ints, against 21 ms for NumPy **[verified]**).

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Status         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The contract is split in two classes with a hole.** `BaseReducer` declares seven abstract methods; `BaseBasicReducer` implements six of them as a check followed by a private hook (`_max`, ...) and leaves `sort` out, with a 6-line comment explaining why `sort` is the exception. A backend written against `BaseReducer` gets none of the checks; a backend written against `BaseBasicReducer` has to know that `sort` is special and write its own iterator check. The docstring example of `BaseReducer` is an instance of a backend.                | **[reading]**  |
| F2  | **The input rule is not shared.** `NativeReducer` takes anything with a length: a set works (`max({1, 2, 3})` is `3`), a NumPy array or a tensor raises `ValueError: The truth value of an array with more than one element is ambiguous` (from `not values` in `_is_empty`). NumPy takes a set and fails with `AttributeError: 'set' object has no attribute 'item'`. torch fails with `RuntimeError: Could not infer dtype of set`. Only generators have a shared, explicit `TypeError`.                                                                       | **[verified]** |
| F3  | **Dimensions are silently flattened.** `NumpyReducer` and `TorchReducer` return `max([[1, 2], [3, 4]])` = `4`, `mean` = `2.5`, `sort` = `[[1, 2], [3, 4]]` (unchanged, not a sorted list of numbers), while `NativeReducer` raises `TypeError`. A 2-D input is almost always a bug of the caller; the answer depends on the backend that `auto_reducer` happened to pick.                                                                                                                                                                                    | **[verified]** |
| F4  | **Errors leak from the library.** The same wrong input gives `AttributeError` (NumPy: big int, `Decimal`, `Fraction`), `ValueError: Overflow when unpacking long long` (torch: `max([2**70, 1])`), `RuntimeError` (torch: `Decimal`, `None`), `UFuncTypeError` (NumPy: strings), `ValueError: too many dimensions 'str'` (torch). `quantiles=[2]` is a `ValueError` in the native and NumPy reducers and a `RuntimeError` in torch; `quantiles=[nan]` the same; `quantile([True, False], ...)` is a `TypeError` in NumPy and works in the other two.            | **[verified]** |
| F5  | **Complex numbers are accepted by one backend.** `NumpyReducer.mean([1+2j, 3])` returns `(2+1j)`, a `complex` for a method documented to return a `float`.                                                                                                                                                                                                                                                                                                                                                                                                 | **[verified]** |
| F6  | **`auto_reducer` picks the slowest backend.** The rule "torch, then NumPy, then native" is applied to a plain list: `TorchReducer` is 2 to 5 times slower than NumPy on a list of one million numbers (`max` 110 ms against 21 ms, `median` 113 ms against 22 ms, `sort` 179 ms against 70 ms) and costs 0.94 s of import when torch is not yet loaded. A call on a short list costs 8.7 µs with torch and 2.8 µs with NumPy or native. The docstring documents the limits (slow, fails beyond 64 bits) of the rule instead of fixing the rule.                  | **[verified]** |
| F7  | **Duplicated rules.** The overflow rule of the median (halve the values) is written three times (`native.py`, `numpy.py`, `torch.py`), the NaN rule of `max`/`min`/`median`/`quantile` twice (native, torch), `_to_tensor` and `_to_float_tensor` are two converters with the same `isinstance(values, (list, tuple))` branch, and `utils.py` imports `BaseReducer` twice (runtime and under `TYPE_CHECKING`) from the package that is being built (`from coola.reducer import BaseReducer, NativeReducer` inside `coola.reducer.utils`).                       | **[reading]**  |
| F8  | **`NativeReducer.mean` is slow.** `statistics.mean` sums exactly with fractions: 104 ms for one million ints, against 6.8 ms for `statistics.fmean` (15 times) and 21 ms for NumPy. `fmean` is accurate to the last bit for the sum (`fsum`), only the final division is rounded, and it returns the same value on the probe (`0.2`).                                                                                                                                                                                                                         | **[verified]** |
| F9  | **`EmptySequenceError` is a bare `Exception`.** `except ValueError` does not catch it, unlike `statistics.StatisticsError` (a `ValueError`) and `max([])` (a `ValueError`) that it replaces. Generic code that handles bad input has to know this one class.                                                                                                                                                                                                                                                                                                | **[reading]**  |
| F10 | **The reducers have no identity.** `NativeReducer() == NativeReducer()` is `False` (default `object` equality) although they are stateless; `InlineDisplayMixin` is imported only to give a `repr`.                                                                                                                                                                                                                                                                                                                                                         | **[verified]** |
| F11 | **Nothing in `src` uses the package.** `grep` finds `coola.reducer` only in its own modules, the tests and the docs. It is a public utility with no internal consumer, so its contract is defined by the guide and the tests only; this is also why the breaking changes below have no internal dependent to update.                                                                                                                                                                                                                                       | **[verified]** |
| F12 | **The guide is behind the code and weaker than the properties.** The **Properties** table has five rows and none for the input, the result types, the NaN rules, the overflow rule or the options; **Design Principles** contains "Type safety: Generic types ensure type correctness" (not a property of a user), the page states that "Most operations raise `EmptySequenceError`" and then lists all six; the custom reducer example defines `_is_empty`, which is an implementation hook of a base class that disappears. `add_backend.md` describes the same hook. | **[reading]**  |
| F13 | **The type parameter is a module-level constrained `TypeVar` that touches the optional library.** `TypeVar("T", Sequence[float], np.ndarray)` in `numpy.py` forces the `else: from coola.utils.fallback.numpy import numpy as np` branch (`# pragma: no cover`) to exist only so that the annotation can be evaluated.                                                                                                                                                                                                                                      | **[reading]**  |

What is already right and must not regress: the lazy exports of `NumpyReducer` and `TorchReducer`;
the NaN behaviour (`max`, `min`, `median`, `quantile` give `NaN` whatever the position, `sort` puts
`NaN` last, like NumPy and torch); the median that does not overflow; the median of an odd number of
ints that stays an `int`; `std` of one value is `NaN`; infinite values make `std` `NaN`;
`sort([])` is `[]`; the generator is rejected with a `TypeError` that says to convert it to a list;
the `to_list` of results (Python numbers); the cross-backend consistency tests and the 46 property
tests (`tests/properties/test_reducer.py`).

## 3. Architecture

```
reducer.max(values)                                      coola/reducer/base.py
        |
        |-- values  = self._prepare(values)              backend: convert, validate, no copy if possible
        |       raises TypeError (iterator, set, str, unsupported element)
        |       raises ValueError (not 1-D)
        |-- len(values) == 0   -> EmptySequenceError     base: one rule, one message
        |-- return self._max(values)                     backend: compute, Python scalar
```

`BaseReducer` is a template: the public methods are concrete, they validate and delegate to the
private computations. A backend implements `_prepare` and six computations. It never sees an empty
input, an iterator, an invalid quantile or the wrong number of dimensions, so it **cannot** forget
one of them, and the contract test can be written once against the base class.

The split is the one of `equality` (the engine owns the invariants, a plugin describes one type),
applied to a package that has no registry: the dispatch is the choice of the instance by the user or
by `auto_reducer`, and there is nothing to register.

### 3.1 The base class

```python
T = TypeVar("T")  # the input type of the backend (what _prepare accepts and returns)


class EmptySequenceError(ValueError):
    """Raised when a statistic is asked on an empty input."""


class BaseReducer(ABC, Generic[T]):
    # Public contract (concrete, not overridden by a backend)
    def max(self, values: T) -> int | float:
        return self._max(self._prepare_non_empty(values, "the maximum"))

    def mean(self, values: T) -> float: ...
    def median(self, values: T) -> int | float: ...
    def min(self, values: T) -> int | float: ...
    def std(self, values: T) -> float: ...

    def quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
        _check_quantiles(quantiles)  # ValueError first
        return self._quantile(self._prepare_non_empty(values, "the quantiles"), quantiles)

    def sort(self, values: T, *, descending: bool = False) -> list[int | float]:
        return self._sort(self._prepare(values), descending=descending)  # no empty check

    # Backend contract
    @abstractmethod
    def _prepare(self, values: T) -> T:
        """Return the values as the 1-D container that the computations use."""

    @abstractmethod
    def _max(self, values: T) -> int | float: ...
    # _mean, _median, _min, _quantile, _sort, _std
```

Rules of the contract (documented in `BaseReducer`, checked by one parametrised unit test over the
three backends, like `NATIVE_SAMPLES` in `equality`):

1. `_prepare` returns a container that supports `len()` and iteration, holds the values in a single
   dimension, and shares memory with the input when it can (it never modifies the input). It raises
   `TypeError` for an iterator (it would be consumed), a set, a `str` / `bytes` or an element the
   backend cannot represent, and `ValueError` for a 0-d or n-d array. The messages are built by two
   helpers of the base class (`_iterator_error`, `_ndim_error`) so that they are identical.
2. `_max` ... `_std` receive a prepared, **non-empty** container and return a Python `int` or
   `float` (`.item()`, `.tolist()`).
3. `_quantile` receives quantiles that are validated: finite floats in `[0, 1]`.
4. `_sort` receives a prepared container that can be empty and returns a new `list`.
5. A backend keeps no state and writes no `__init__` unless it checks that its library is installed.

`descending` becomes keyword-only (`CLAUDE.md`: keep new options keyword-only; `reducer.sort(v, True)`
is not readable). `EmptySequenceError` derives from `ValueError` (F9).

### 3.2 The backends

| Backend         | `_prepare`                                                                                                                                                                                                                                                       | Computation                                                                                                      |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `NativeReducer` | Accept a `collections.abc.Sequence` that is not a `str` / `bytes`; reject everything else (arrays and tensors included: `TypeError("NativeReducer takes a list or a tuple, got ndarray; use NumpyReducer")`). Returns the input itself, no copy.                    | `max`, `min`, `statistics.fmean` (F8), `median` with the halving rule, `stdev`, `utils.stats.quantile`, `sorted`. |
| `NumpyReducer`  | `np.asarray(values)`; reject a `dtype` that is not `bool` / int / uint / float (`object` for a big int or a `Decimal`, complex, string) with a `TypeError` naming `NativeReducer`; reject `ndim != 1`. Rejects a set and an iterator before `asarray` (F2).        | `np.max`, `np.min`, `np.mean`, `np.median` (halving rule), `np.std(ddof=1)`, `np.quantile`, `np.sort`.            |
| `TorchReducer`  | Tensor: as is. List / tuple: one converter `_to_tensor(values)` (float64 if any float, else int64; `OverflowError` / `ValueError` of the conversion become the same `TypeError`). Reject complex and `ndim != 1`. An array goes through `coola.utils.tensor.to_tensor`. | `torch.max`, `torch.min`, `mean` on float64, `median` of two middle values, `std`, `quantile`, `sort`.           |

The halving rule of the median ("if the mean of the two middle values overflowed, halve the values
first") is the same sentence in the three backends. It stays in the backends because it needs the
backend's own `median` and `isfinite`; the contract test (section 7) pins it.

### 3.3 `auto_reducer`

```python
def auto_reducer() -> BaseReducer:
    if is_numpy_available():
        return NumpyReducer()
    return NativeReducer()
```

NumPy first, then native; `TorchReducer` is **never** chosen implicitly (F6). The two reasons are
measured: on a plain list it is the slowest of the three, and it imports torch (0.94 s). A user who
has tensors on a device builds `TorchReducer()` explicitly; a user with CPU tensors can pass them to
`NumpyReducer` (`np.asarray(tensor)` works without a copy for a CPU tensor without gradient). The
function stays a function without an argument: choosing the backend from the type of the values
would be a second dispatch mechanism for a package that has three classes (decision D3).

`auto_reducer` stays in `utils.py`, but imports `BaseReducer` and `NativeReducer` from their modules
(`coola.reducer.base`, `coola.reducer.native`), not from the package that is being built (F7).

## 4. Public API after the change

```python
from coola.reducer import (
    BaseReducer,  # template; BaseBasicReducer is removed
    EmptySequenceError,  # now a ValueError
    NativeReducer,
    NumpyReducer,  # lazy
    TorchReducer,  # lazy
    auto_reducer,  # () -> BaseReducer; NumPy, then native
)
```

| Change                             | Before                                        | After                                                                    |
| ---------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------ |
| `BaseBasicReducer`                 | exported, with `_is_empty` and six hooks      | removed; `BaseReducer` has `_prepare` and `_max` ... `_sort`, `_std`     |
| `sort(values, descending)`         | positional or keyword                         | `sort(values, *, descending=False)`                                      |
| `EmptySequenceError`               | `Exception`                                   | `ValueError`                                                             |
| Input                              | anything the backend happens to accept        | a sequence of reals, or a 1-D array / tensor for the backend (rule 1)    |
| Wrong input                        | `AttributeError`, `RuntimeError`, `ValueError` | `TypeError` (type / element), `ValueError` (shape), same text            |
| `quantile` with an invalid `q`     | `ValueError` or `RuntimeError`                | `ValueError("quantiles must be in [0, 1], got 2")` in the three backends |
| `auto_reducer()` with both libs    | `TorchReducer`                                | `NumpyReducer`                                                           |
| `NativeReducer.mean`               | exact (`statistics.mean`)                     | `statistics.fmean` (15 times faster; the last bit can differ for floats) |
| Reducer equality                   | identity                                      | unchanged (see D4)                                                       |

## 5. Dependents

| Package / file                                 | Change                                                                                                                                                                                 |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src` (everything except `coola.reducer`)      | None: no module imports `coola.reducer` (F11). `coola.utils.stats.quantile` is used only by `NativeReducer`; it stays in `coola.utils` (public helper with its own tests, `CLAUDE.md`). |
| `docs/docs/uguide/reducer.md`                  | Rewritten around the properties of section 1; the custom reducer example implements `_prepare` and the six computations; `auto_reducer` priority.                                      |
| `docs/docs/dev/add_backend.md` (line 64)       | "add a `BaseBasicReducer`" becomes "subclass `BaseReducer` (`_prepare` + six computations)"; the consistency test is now the contract test.                                           |
| `docs/docs/dev/architecture.md` (lines 537-538) | No change: `reducer` stays outside `registry`.                                                                                                                                         |
| `docs/docs/migration/v0.11_to_v1.0.md`         | New section: `BaseBasicReducer`, `_is_empty`, `sort(descending=...)`, `EmptySequenceError`, `auto_reducer`, the input rule.                                                            |
| `tests/unit/reducer/`, `tests/properties/`     | Ported (section 7); the 46 properties keep their meaning.                                                                                                                              |

## 6. Evidence

All numbers: Apple M1, Python 3.14, NumPy and torch installed, one run, list of one million ints
unless said otherwise. **[verified]**

| Probe                                  | `NativeReducer` | `NumpyReducer` | `TorchReducer` |
| -------------------------------------- | --------------- | -------------- | -------------- |
| `mean` (list, 10^6)                    | 104.1 ms        | 21.4 ms        | 46.4 ms        |
| `median`                               | 66.7 ms         | 22.0 ms        | 112.6 ms       |
| `max`                                  | 43.2 ms         | 21.4 ms        | 110.2 ms       |
| `sort`                                 | 101.1 ms        | 69.7 ms        | 179.0 ms       |
| `mean([1, 2, 3])` (per call)           | 2.9 µs          | 2.8 µs         | 8.7 µs         |
| `statistics.fmean` (list, 10^6)        | 6.8 ms          |                |                |
| `import torch`                         |                 |                | 0.94 s         |

Behaviour probes (same input to the three backends, the table lists only the rows that differ):

| Input                      | Native                       | NumPy                                 | torch                                 |
| -------------------------- | ---------------------------- | ------------------------------------- | ------------------------------------- |
| `max({1, 2, 3})`           | `3`                          | `AttributeError`                      | `RuntimeError`                        |
| `max(np.array([[1,2],[3,4]]))` | `ValueError` (ambiguous)  | `4`                                   | `ValueError` (ambiguous)              |
| `max([[1, 2], [3, 4]])`    | `TypeError`                  | `4`                                   | `4`                                   |
| `max([2**70, 1])`          | `1180591620717411303424`     | `AttributeError`                      | `ValueError` (overflow)               |
| `mean([1+2j, 3])`          | `TypeError`                  | `(2+1j)`                              | `TypeError`                           |
| `quantile([1,2,3], [2])`   | `ValueError`                 | `ValueError` (other message)          | `RuntimeError`                        |
| `quantile([True, False], [0, .5, 1])` | `[0.0, 1.0, 1.0]` | `TypeError`                           | `[0.0, 1.0, 1.0]`                     |
| `mean([1e308, 1e308])`     | `1e308`                      | `inf`                                 | `inf`                                 |
| `median([True, False, True])` | `True`                    | `1.0`                                 | `True`                                |

Not yet measured, to do in step 2 before the code is written: the cost of `_prepare` on a list for
NumPy (`np.asarray` runs once instead of once per call: no change), and `fmean` against
`statistics.mean` on floats with large cancellation (the property tests compare with a tolerance).

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                  | Breaking                  |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| 1    | **Guide (F12).** Put the **Properties** of section 1 at the top of `uguide/reducer.md` and in the package docstring, remove "Type safety: Generic types", fix "Most operations raise", state the input rule, the result types and the "not guaranteed" list. Documentation only.                                                                                                                                                           | no                        |
| 2    | **`auto_reducer` (F6, F7).** NumPy, then native; imports from the modules; docstring rewritten; a benchmark file `tests/benchmarks/test_reducer_benchmark.py` (mean, median, sort on a list of 100,000 and `mean([1, 2, 3])` for the three backends) that documents the numbers of section 6.                                                                                                                                              | yes (backend choice)      |
| 3    | **`BaseReducer` is the template (F1, F9, F13).** Merge `BaseBasicReducer` into `BaseReducer` (`_prepare`, private computations, `sort` goes through `_prepare` only), `EmptySequenceError(ValueError)`, `sort(..., *, descending=False)`, the quantile validation, the shared error helpers. Port the three backends with their current `_prepare` (a pass-through that keeps today's behaviour). Delete `_is_empty`, `_check_values`, the `T` `TypeVar` with `np.ndarray`. | yes (custom reducers)     |
| 4    | **Input rule and errors (F2, F3, F4, F5).** The real `_prepare` of the three backends (sequence only for native; `dtype` and `ndim` checks for NumPy and torch; set / iterator / `str` rejected; `OverflowError` and `ValueError` of the torch conversion translated; one `_to_tensor`). Regression tests for every row of the behaviour table that changes.                                                                         | yes (errors)              |
| 5    | **`NativeReducer.mean` with `fmean` (F8)** and removal of the copies of the median rule that the base can own after step 3 (only if the code is shorter; otherwise keep, section 3.2). Benchmark gate.                                                                                                                                                                                                                                    | no (last bit of floats)   |
| 6    | **Documentation.** `uguide/reducer.md` (custom reducer example), `dev/add_backend.md`, the migration page, the status table of `review/suggestions.md` with the PR numbers.                                                                                                                                                                                                                                                              | no                        |

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/reducer/`); the conventions of `CLAUDE.md` apply (one
behaviour per test, literal expected values, a regression test for every bug that is fixed and a
check that it fails on `main`, no local imports, `*_available` fixtures).

- **Contract test over the three backends** (`test_consistency.py`, rewritten as the contract of
  `BaseReducer`, one parametrised fixture `reducer`, as today):
  - the typical result of each method, as a literal (`max` = `2`, `mean` = `0.0`, ...), and the
    **type** of the result with `type(result) is int` / `is float` (property "Python results");
  - the output of `quantile` and `sort` against a fully-specified literal list;
  - empty input for the six statistics: `EmptySequenceError`, which is also caught as `ValueError`,
    with the same message in every backend; `sort([])` is `[]`; `quantile([], [])` raises;
  - the input rule, one test per row: a generator, a set, a `str`, a 2-D array, a 0-d array, a big
    int, a complex number, `None` in the list: the same exception type and message in every backend
    (for the cases the backend can express);
  - the options: `quantiles` = `[-0.1]`, `[1.1]`, `[nan]`: `ValueError` with the message of the base
    class and **before** the empty check (`quantile([], [2])` raises `ValueError`);
  - the NaN rules in every position (first, middle, last), the infinite `std`, the median that does not
    overflow, `std` of one value;
  - purity: the input list / array / tensor is equal after the call; the call is repeatable;
  - `descending` is keyword-only (`pytest.raises(TypeError)` for the positional call).
- **`BaseReducer` itself** (`test_base.py`, new): a minimal backend that records the calls proves that
  `_max` is never called with an empty input, that `_prepare` is called exactly once per call and
  before the empty check, that `_sort` is called for an empty input, and that the quantiles are
  validated before `_prepare` (the "valid options first" property).
- **Each backend** (`test_native.py`, `test_numpy.py`, `test_torch.py`): what is specific to it only:
  the `dtype` / `device` handling, `MissingPackageError`, the input types it accepts beyond lists
  (arrays for NumPy; tensors of all integer and float dtypes, a CPU tensor with `requires_grad`
  for torch), the translation of the library's errors (`Decimal`, big int), the repr.
- **`auto_reducer`** (`test_utils.py`): patched availability gives NumPy, then native (unit);
  **integration** test without patching: the real `auto_reducer()` returns a reducer whose result on
  a list equals `NativeReducer`'s (CLAUDE.md: a patched test needs a real one).
- **Properties** (`tests/properties/test_reducer.py`, keep the 46, add): the output of every backend is
  within a tolerance of `NativeReducer` for random float lists (kept); `sort` is a permutation and is
  ordered (kept); a quantile is between `min` and `max` and is monotone in `q` (new); `mean` is between
  `min` and `max` (new, finite inputs); the result type is `int | float` for any generated list (new).
- **Integration** (`tests/integration/`): threads sharing one reducer instance call every method with
  a lock around the shared list (`run_threads`, docstring "Test that ..."); `import coola.reducer`
  does not import `numpy` or `torch` (a subprocess, `sys.modules`).
- **Doctests** run with the docstrings; every example of the guide is a doctest.
- **Benchmark gate** (`tests/benchmarks/test_reducer_benchmark.py`): the numbers of section 6; the new
  `NativeReducer.mean` must not be slower than `main`, and `auto_reducer().mean` must not be slower
  than `TorchReducer().mean` on a list of 100,000 (expected: faster).
- Coverage: 100 % of `src/coola/reducer` (`--cov=coola`), no `# pragma: no cover` except the
  `fallback` import branch if it is kept (it is not needed once F13 is fixed).

## 9. Decisions for the maintainers

**D1. One base class or two?** _Recommended: one (`BaseReducer` as a template)._ Today the user must
choose between a class that carries no checks and a class that carries them for six methods out of
seven (F1). Nobody needs the first: a backend that wants to skip the checks overrides the public
method. The price is that a custom reducer is ported once (the migration page). Alternative: keep
both and fix `sort` in `BaseBasicReducer`; less breaking, but the empty-case property still depends
on which class the author picked.

**D2. How strict is the input?** _Recommended: strict (sequence of reals, or 1-D array / tensor of the
backend; everything else is an error with the same message)._ A silent flatten of a 2-D array (F3)
gives a number that looks right and is not; a set is accepted by one backend and not the others
(F2). Alternative: be liberal, convert everything with `np.asarray` / `torch.as_tensor` and
document the differences; shorter code, but the same call changes meaning with `auto_reducer`. The
strictness costs one `isinstance` and one `ndim` check per call (nanoseconds against a 2.8 µs call).

**D3. Does `auto_reducer` look at the values?** _Recommended: no._ `auto_reducer()` returns NumPy,
then native; a tensor on a device is a reason to write `TorchReducer()`. A signature
`auto_reducer(values)` that returns `TorchReducer` for a tensor, `NumpyReducer` for an array and the
default otherwise is about ten lines and a second dispatch; it can be added later without breaking
anything, once a consumer asks. Alternative: keep "torch first" so that tensors are always handled;
rejected on the measurements of F6 (slower on lists, 0.94 s of import).

**D4. `__eq__` and `__hash__` for reducers?** _Recommended: no._ A reducer is stateless but the user
rarely compares two of them; the `equal` methods that the summary design deletes (`review/summary.md`,
F3) carry no information either. If an `__eq__` is wanted (for example to cache a function by its
reducer), `type(a) is type(b)` is one line for the base class and does not change the design.
`InlineDisplayMixin` (the `repr`) stays: it costs nothing.

**D5. Keep `TorchReducer` at all?** _Recommended: yes._ It is the only backend for a tensor that lives
on a GPU, and tensors are the data of the users of the sibling packages (`equality`, `summary`
handle them). It is also the most expensive to maintain (129 lines, two converters, a
median that is not the library's). The step 4 cleanup (one converter, the shared errors) brings it
down; dropping it would be a decision about the scope of `coola`, not about this package, and
nothing in `src` depends on it (F11).

**D6. `NativeReducer.mean` exact or fast?** _Recommended: `statistics.fmean`._ The exact mean of
`statistics.mean` is irrelevant to the cross-backend tolerance (NumPy and torch already differ in the
last bit), and the time (104 ms against 6.8 ms) is the largest gap of the package. `fmean` also
returns a `float` for ints, which is the documented result type. Alternative: keep `mean` and accept
that the native reducer is 5 times slower than NumPy for the one method that is used the most.

## 10. Not covered

- **More statistics** (`variance`, `mode`, `sum`, `argmax`, `percentile`, `histogram`): the six methods
  and `sort` are the contract. Each new statistic is one public method of `BaseReducer`, one
  abstract computation, three implementations and a contract test: this is why none is added without
  a consumer (F11).
- **Axis / multi-dimensional reductions, weights, `ddof`, `skipna`**: out of scope; the package reduces
  one sequence. `std` is the sample standard deviation (`ddof=1`) in the three backends.
- **A registry of reducers** (`register`, lazy loading by type name, plugin entry point): three
  classes do not justify it (`architecture.md` lists `reducer` outside `registry` on purpose). If
  `auto_reducer(values)` (D3) ever dispatches on more than two types, it is the moment to reuse
  `BaseTypeDispatchRegistry`.
- **Other backends** (pandas `Series`, polars, JAX): `add_backend.md` explains how to add one; the
  contract test is the acceptance test.
- **Lazy / streaming input** (`statistics` of an iterator in one pass): the iterator is rejected on
  purpose (the reducers go over the values several times); the error says to convert it to a list.
- **Precision beyond float64** (`Decimal`, `Fraction`): `NativeReducer` happens to work; it is not a
  property.

## Appendix: effect on the size of the package

Estimates from reading the code to delete; no prototype was written. The line counts include
docstrings, which are most of `base.py`.

| Module      | Lines today | Expected after | Why                                                                                                                  |
| ----------- | ----------- | -------------- | -------------------------------------------------------------------------------------------------------------------- |
| `base.py`   | 342         | about 290      | one class, the six public methods with their docstrings and examples, the shared checks; no `_is_empty`, no 6-line comment on `sort` |
| `native.py` | 86          | about 80       | `_prepare` added, `fmean`, `_is_empty` and the module `TypeVar` removed                                                 |
| `numpy.py`  | 89          | about 90       | `_prepare` with the `dtype` / `ndim` check replaces `_is_empty`; no `fallback` import branch                          |
| `torch.py`  | 129         | about 110      | one `_to_tensor`, `_prepare` replaces `_is_empty`                                                                       |
| `utils.py`  | 47          | about 35       | two lines instead of three branches, no double import                                                                   |
| `__init__`  | 32          | about 30       | `BaseBasicReducer` removed                                                                                              |
| Tests       | 2,190       | about 2,000    | the contract test replaces the per-backend copies of the empty, iterator and NaN tests; the new rows add some back     |

The gate is the behaviour (sections 1 and 8), not the size.
