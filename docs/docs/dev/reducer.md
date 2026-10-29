# Architecture of `coola.reducer`

:book: This page describes the design of the `coola.reducer` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/reducer.md#properties); the API is in the [reference](../refs/reducer.md).

## Constraints

1. **Simple and easy to maintain.** The package is 725 lines of source and 2,190 lines of tests. The
   target removes the code that exists only because the contract is spread over the backends (the
   `BaseReducer` / `BaseBasicReducer` pair with a hole for `sort`, `_is_empty` and the `_check_values`
   hook, the copies of the overflow rule of the median, two tensor converters) instead of adding code
   next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. A custom reducer written for the current hooks is ported (the
   migration page lists the change).
3. **No slower than before**, and faster where a probe shows an easy win (the mean of `NativeReducer`
   is 15 times slower than `statistics.fmean`, ). A design that is simpler but slower is rejected.
4. **Consistent with the siblings.** Keyword-only options, strict contracts (an invalid option or
   input raises before any work, with a message that names the remedy), stateless objects that can be
   shared by threads, lazy optional libraries.
5. **Optional libraries stay lazy.** `import coola.reducer` imports neither NumPy nor PyTorch
   (`lazy_exports`, as before). `TorchReducer` costs 0.94 s of import, which is why
   `auto_reducer` must not choose it by default.

## Architecture

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

### The base class

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
is not readable). `EmptySequenceError` derives from `ValueError`.

### The backends

| Backend         | `_prepare`                                                                                                                                                                                                                                                              | Computation                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `NativeReducer` | Accept a `collections.abc.Sequence` that is not a `str` / `bytes`; reject everything else (arrays and tensors included: `TypeError("NativeReducer takes a list or a tuple, got ndarray; use NumpyReducer")`). Returns the input itself, no copy.                        | `max`, `min`, `statistics.fmean`, `median` with the halving rule, `stdev`, `utils.stats.quantile`, `sorted`. |
| `NumpyReducer`  | `np.asarray(values)`; reject a `dtype` that is not `bool` / int / uint / float (`object` for a big int or a `Decimal`, complex, string) with a `TypeError` naming `NativeReducer`; reject `ndim != 1`. Rejects a set and an iterator before `asarray`.                  | `np.max`, `np.min`, `np.mean`, `np.median` (halving rule), `np.std(ddof=1)`, `np.quantile`, `np.sort`.       |
| `TorchReducer`  | Tensor: as is. List / tuple: one converter `_to_tensor(values)` (float64 if any float, else int64; `OverflowError` / `ValueError` of the conversion become the same `TypeError`). Reject complex and `ndim != 1`. An array goes through `coola.utils.tensor.to_tensor`. | `torch.max`, `torch.min`, `mean` on float64, `median` of two middle values, `std`, `quantile`, `sort`.       |

The halving rule of the median ("if the mean of the two middle values overflowed, halve the values
first") is the same sentence in the three backends. It stays in the backends because it needs the
backend's own `median` and `isfinite`; the contract test pins it.

### `auto_reducer`

```python
def auto_reducer() -> BaseReducer:
    if is_numpy_available():
        return NumpyReducer()
    return NativeReducer()
```

NumPy first, then native; `TorchReducer` is **never** chosen implicitly. The two reasons are
measured: on a plain list it is the slowest of the three, and it imports torch (0.94 s). A user who
has tensors on a device builds `TorchReducer()` explicitly; a user with CPU tensors can pass them to
`NumpyReducer` (`np.asarray(tensor)` works without a copy for a CPU tensor without gradient). The
function stays a function without an argument: choosing the backend from the type of the values
would be a second dispatch mechanism for a package that has three classes.

`auto_reducer` stays in `utils.py`, but imports `BaseReducer` and `NativeReducer` from their modules
(`coola.reducer.base`, `coola.reducer.native`), not from the package that is being built.

## Public API

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

| Change                          | Before                                         | After                                                                    |
| ------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| `BaseBasicReducer`              | exported, with `_is_empty` and six hooks       | removed; `BaseReducer` has `_prepare` and `_max` ... `_sort`, `_std`     |
| `sort(values, descending)`      | positional or keyword                          | `sort(values, *, descending=False)`                                      |
| `EmptySequenceError`            | `Exception`                                    | `ValueError`                                                             |
| Input                           | anything the backend happens to accept         | a sequence of reals, or a 1-D array / tensor for the backend (rule 1)    |
| Wrong input                     | `AttributeError`, `RuntimeError`, `ValueError` | `TypeError` (type / element), `ValueError` (shape), same text            |
| `quantile` with an invalid `q`  | `ValueError` or `RuntimeError`                 | `ValueError("quantiles must be in [0, 1], got 2")` in the three backends |
| `auto_reducer()` with both libs | `TorchReducer`                                 | `NumpyReducer`                                                           |
| `NativeReducer.mean`            | exact (`statistics.mean`)                      | `statistics.fmean` (15 times faster; the last bit can differ for floats) |
| Reducer equality                | identity                                       | unchanged                                                                |

## Dependents

| Package / file                             | Change                                                                                                                                                                            |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src` (everything except `coola.reducer`)  | None: no module imports `coola.reducer`. `coola.utils.stats.quantile` is used only by `NativeReducer`; it stays in `coola.utils` (public helper with its own tests, `CLAUDE.md`). |
| `tests/unit/reducer/`, `tests/properties/` | Ported; the 46 properties keep their meaning.                                                                                                                                     |

## Decisions

All six decisions are made and follow the recommendations, so the design of stands
unchanged.

| Decision | Question                                | Outcome                           | Where it applies               |
| -------- | --------------------------------------- | --------------------------------- | ------------------------------ |
| D1       | One base class or two?                  | One (`BaseReducer` as a template) | Section 3.1                    |
| D2       | How strict is the input?                | Strict                            | Section 1 (input rule)         |
| D3       | Does `auto_reducer` look at the values? | No                                | Section 3.3                    |
| D4       | `__eq__` and `__hash__` for reducers?   | No                                | Section 4 (equality unchanged) |
| D5       | Keep `TorchReducer`?                    | Yes                               | Section 3.2                    |
| D6       | `NativeReducer.mean` exact or fast?     | `statistics.fmean`                | Section 4                      |

The reasoning for each is kept below.

**D1. One base class or two?** _Recommended: one (`BaseReducer` as a template)._ Today the user must
choose between a class that carries no checks and a class that carries them for six methods out of
seven. Nobody needs the first: a backend that wants to skip the checks overrides the public
method. The price is that a custom reducer is ported once (the migration page). Alternative: keep
both and fix `sort` in `BaseBasicReducer`; less breaking, but the empty-case property still depends
on which class the author picked.

Decision: one

**D2. How strict is the input?** _Recommended: strict (sequence of reals, or 1-D array / tensor of the
backend; everything else is an error with the same message)._ A silent flatten of a 2-D array
gives a number that looks right and is not; a set is accepted by one backend and not the others
. Alternative: be liberal, convert everything with `np.asarray` / `torch.as_tensor` and
document the differences; shorter code, but the same call changes meaning with `auto_reducer`. The
strictness costs one `isinstance` and one `ndim` check per call (nanoseconds against a 2.8 µs call).

Decision: strict

**D3. Does `auto_reducer` look at the values?** _Recommended: no._ `auto_reducer()` returns NumPy,
then native; a tensor on a device is a reason to write `TorchReducer()`. A signature
`auto_reducer(values)` that returns `TorchReducer` for a tensor, `NumpyReducer` for an array and the
default otherwise is about ten lines and a second dispatch; it can be added later without breaking
anything, once a consumer asks. Alternative: keep "torch first" so that tensors are always handled;
rejected on the measurements of (slower on lists, 0.94 s of import).

Decision: no

**D4. `__eq__` and `__hash__` for reducers?** _Recommended: no._ A reducer is stateless but the user
rarely compares two of them; the `equal` methods that the summary design deletes (the `summary` design,
) carry no information either. If an `__eq__` is wanted (for example to cache a function by its
reducer), `type(a) is type(b)` is one line for the base class and does not change the design.
`InlineDisplayMixin` (the `repr`) stays: it costs nothing.

Decision: no

**D5. Keep `TorchReducer` at all?** _Recommended: yes._ It is the only backend for a tensor that lives
on a GPU, and tensors are the data of the users of the sibling packages (`equality`, `summary`
handle them). It is also the most expensive to maintain (129 lines, two converters, a
median that is not the library's). The cleanup (one converter, the shared errors) brings it
down; dropping it would be a decision about the scope of `coola`, not about this package, and
nothing in `src` depends on it.

Decision: yes

**D6. `NativeReducer.mean` exact or fast?** _Recommended: `statistics.fmean`._ The exact mean of
`statistics.mean` is irrelevant to the cross-backend tolerance (NumPy and torch already differ in the
last bit), and the time (104 ms against 6.8 ms) is the largest gap of the package. `fmean` also
returns a `float` for ints, which is the documented result type. Alternative: keep `mean` and accept
that the native reducer is 5 times slower than NumPy for the one method that is used the most.

Decision: fmean

## Not covered

- **More statistics** (`variance`, `mode`, `sum`, `argmax`, `percentile`, `histogram`): the six methods
  and `sort` are the contract. Each new statistic is one public method of `BaseReducer`, one
  abstract computation, three implementations and a contract test: this is why none is added without
  a consumer.
- **Axis / multi-dimensional reductions, weights, `ddof`, `skipna`**: out of scope; the package reduces
  one sequence. `std` is the sample standard deviation (`ddof=1`) in the three backends.
- **A registry of reducers** (`register`, lazy loading by type name, plugin entry point): three
  classes do not justify it (`architecture.md` lists `reducer` outside `registry` on purpose). If
  `auto_reducer(values)` ever dispatches on more than two types, it is the moment to reuse
  `BaseTypeDispatchRegistry`.
- **Other backends** (pandas `Series`, polars, JAX): `add_backend.md` explains how to add one; the
  contract test is the acceptance test.
- **Lazy / streaming input** (`statistics` of an iterator in one pass): the iterator is rejected on
  purpose (the reducers go over the values several times); the error says to convert it to a list.
- **Precision beyond float64** (`Decimal`, `Fraction`): `NativeReducer` happens to work; it is not a
  property.
