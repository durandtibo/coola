# Architecture of `coola.reducer`

:book: This page describes the design of the `coola.reducer` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/reducer.md#properties); the API is in the [reference](../refs/reducer.md).

## Constraints

1. **Simple and easy to maintain.** The contract is written once, in the base class. A backend
   implements `_prepare` and six computations; the empty check, the quantile check and the dimension
   check do not exist in the backends, and the overflow rule of the median is stated in one place.
2. **Fast.** The mean of `NativeReducer` uses `statistics.fmean`, which is much faster than
   `statistics.mean` (104 ms against 6.8 ms in a probe). A design that is simpler but slower is
   rejected.
3. **Consistent with the siblings.** Keyword-only options, strict contracts (an invalid option or
   input raises before any work, with a message that names the remedy), stateless objects that can be
   shared by threads, lazy optional libraries.
4. **Optional libraries stay lazy.** `import coola.reducer` imports neither NumPy nor PyTorch
   (`lazy_exports`). `TorchReducer` costs 0.94 s of import, which is why
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

`descending` is keyword-only (`reducer.sort(v, True)` is not readable). `EmptySequenceError` derives
from `ValueError`.

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
    BaseReducer,  # template
    EmptySequenceError,  # a ValueError
    NativeReducer,
    NumpyReducer,  # lazy
    TorchReducer,  # lazy
    auto_reducer,  # () -> BaseReducer; NumPy, then native
)
```

| Behaviour                       | Rule                                                                     |
| ------------------------------- | ------------------------------------------------------------------------ |
| Input                           | a sequence of reals, or a 1-D array / tensor for the backend (rule 1)    |
| Wrong input                     | `TypeError` (type / element), `ValueError` (shape), same text            |
| `quantile` with an invalid `q`  | `ValueError("quantiles must be in [0, 1], got 2")` in the three backends |
| `auto_reducer()` with both libs | `NumpyReducer`                                                           |
| `NativeReducer.mean`            | `statistics.fmean` (the last bit can differ for floats)                  |
| `sort(values, *, descending)`   | `descending` is keyword-only                                             |
| Reducer equality                | identity                                                                 |

## Dependents

| Package / file                             | Use                                                                                                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src` (everything except `coola.reducer`)  | No module imports `coola.reducer`. `coola.utils.stats.quantile` is used only by `NativeReducer`; it is a public helper of `coola.utils` with its own tests.                |
| `tests/unit/reducer/`, `tests/properties/` | The contract test over the three backends, and the properties of the reducers.                                                                                            |

## Design choices and trade-offs

**One base class, a template.** A pair of base classes (one without checks and one with them) leaves
the author to choose, and the empty-case property then depends on the class that was picked. With one
template, a backend that wants to skip a check has to override the public method on purpose.

**Strict input.** A silent flatten of a 2-D array gives a number that looks right and is not, and a
set is accepted by one backend and not by the others. Being liberal (convert everything with
`np.asarray` / `torch.as_tensor` and document the differences) gives shorter code, but the same call
changes meaning with `auto_reducer`. The strictness costs one `isinstance` and one `ndim` check per
call (nanoseconds against a 2.8 µs call).

**`auto_reducer` does not look at the values.** `auto_reducer()` returns NumPy, then native; a tensor
on a device is a reason to write `TorchReducer()`. An `auto_reducer(values)` that returns
`TorchReducer` for a tensor, `NumpyReducer` for an array and the default otherwise would be about ten
lines and a second dispatch mechanism; it can be added without breaking anything once a consumer asks.
"Torch first" would handle tensors always, but is slower on lists and costs 0.94 s of import.

**No `__eq__` and `__hash__` for reducers.** A reducer is stateless, and two of them are rarely
compared. If an `__eq__` is wanted (for example to cache a function by its reducer), `type(a) is
type(b)` is one line in the base class. `InlineDisplayMixin` (the `repr`) costs nothing and is kept.

**`TorchReducer` stays.** It is the only backend for a tensor that lives on a GPU, and tensors are the
data of the users of the sibling packages (`equality`, `summary`). It is also the most expensive to
maintain (two converters, a median that is not the library's), which is why it has one converter and
the shared errors. Dropping it would be a decision about the scope of `coola`, not about this package.

**`NativeReducer.mean` is fast, not exact.** The exact mean of `statistics.mean` is irrelevant to the
cross-backend tolerance (NumPy and torch already differ in the last bit), and the time gap is the
largest of the package. `fmean` also returns a `float` for ints, which is the documented result type.

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
- **Other backends** (pandas `Series`, polars, JAX): [How to add support for a new library](add_backend.md) explains how to add one; the contract test is
  the acceptance test.
- **Lazy / streaming input** (`statistics` of an iterator in one pass): the iterator is rejected on
  purpose (the reducers go over the values several times); the error says to convert it to a list.
- **Precision beyond float64** (`Decimal`, `Fraction`): `NativeReducer` happens to work; it is not a
  property.
