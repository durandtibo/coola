# Architecture of `coola.random`

:book: This page describes the design of the `coola.random` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/random.md#properties); the API is in the [reference](../refs/random.md).

## Constraints

1. **Simple and easy to maintain.** The package is 1,044 lines of source (1,605 of unit tests) to
   call three seeding functions. About 250 of these lines are wrappers that re-implement
   `coola.registry.Registry` (`RandomManagerRegistry`) and docstrings of two functions that only
   rename `Manager().seed()`. The target removes code that exists only because of the shape of the
   current API, instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period. The release notes list the changes. The one exception is deliberate and
   free: the random numbers produced by a seed that is already accepted before must not change
   (reproducibility of the user's experiments is the whole point of the package, see D2).
3. **Standalone helper.** `coola.random` imports only core packages (`registry`, `utils`,
   `display`). It is not a type-dispatch registry and does not become one (the key of a manager is a
   name, not a type).
4. **Optional libraries stay lazy.** `import coola.random` imports neither NumPy nor torch. The
   managers of the optional libraries are in their own modules, and the default registry imports
   them when it is built.
5. **Process-global by nature.** The generators it manages are process-global. The package does not
   try to make them thread-local or to lock them: it cannot, because other code draws from the
   same generators. It states this limit instead.

## Architecture

```
manual_seed(seed) / random_seed(seed) / get_rng_state() / set_rng_state(state)
        |   module functions: forward to the default registry (or to the registry given)
        v
RandomManagerRegistry                     Registry[str, BaseRandomManager] + BaseRandomManager
        |   key -> manager, in registration order; is itself a manager (composite)
        |   manual_seed : validate once, then every manager
        |   get_rng_state: {key: manager.get_rng_state()}
        |   set_rng_state: same keys required, restore all, then raise
        v
BaseRandomManager (ABC)                   get_rng_state / manual_seed / set_rng_state + seed()
        |   seed(): validate, save, try: seed, yield; finally: restore
        v
PythonRandomManager   NumpyRandomManager   TorchRandomManager   <plugin managers>
   random.seed          np.random.seed       torch.manual_seed
```

The rule of the split: **a manager knows one library; the registry knows the keys; the base class
knows the scoped context.** Nothing else exists. There is no engine, no dispatch by type, and no
state beyond the key to manager mapping.

### Public API

| Name                                            | Role                                                                                                 | Change                                         |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `manual_seed(seed)`                             | Seed every manager of the default registry.                                                          | unchanged                                      |
| `random_seed(seed, *, registry=None)`           | Scoped seed; restores the state of the registry.                                                     | `manager` is now the keyword-only `registry`   |
| `get_rng_state()` / `set_rng_state(state)`      | Save / restore the state of the default registry.                                                    | `set_rng_state` is strict                      |
| `register_managers(mapping, *, exist_ok=False)` | Add managers to the default registry.                                                                | `exist_ok` keyword-only                        |
| `get_default_registry()` and its alias          | The default `RandomManagerRegistry`.                                                                 | unchanged                                      |
| `validate_seed(seed)`                           | Raise `TypeError` / `ValueError` for a seed out of the domain.                                       | unchanged                                      |
| `BaseRandomManager`                             | The extension point: three abstract methods and the concrete `seed()`.                               | `seed()` validates; documented contract        |
| `PythonRandomManager`                           | `random`.                                                                                            | renamed from `RandomRandomManager`             |
| `NumpyRandomManager`, `TorchRandomManager`      | NumPy, torch (lazy exports).                                                                         | validate; NumPy no longer folds; torch simpler |
| `RandomManagerRegistry`                         | A registry of managers that is a manager.                                                            | subclass of `Registry`                         |
| `get_torch_generator(seed, *, device="cpu")`    | A `torch.Generator` seeded with `seed` (lazy export).                                                | exported; `seed`, keyword-only `device`        |
| `get_random_seed(seed)`                         | A deterministic seed in `[0, 2**63 - 1)` derived from `seed`, valid for `manual_seed` (lazy export). | range fixed, exported                          |

Removed: `numpy_seed`, `torch_seed`, `RandomRandomManager`, `RandomManagerRegistry.has_manager`,
`find_manager`, and the warnings of `set_rng_state`.

### The contract of a manager

Documented in `BaseRandomManager`, and checked for every manager by one parametrised unit test
(a `MANAGERS` list built with the `*_available` fixtures, like `NATIVE_SAMPLES` in `equality`):

1. `get_rng_state()` returns an opaque object that does not alias the live generator (drawing a
   number afterwards does not change it).
2. `set_rng_state(get_rng_state())` leaves the draws unchanged.
3. `manual_seed(seed)` validates the seed with `validate_seed` before it touches a generator, and
   the same seed gives the same draws.
4. Two different valid seeds give different draws (injective, ).
5. A manager is stateless and cheap to build: it is created once, shared by registries, equal to
   another instance of its class, and hashable (`__eq__` and `__hash__` on the type).
6. Building a manager of an uninstalled library raises `ImportError` (`check_numpy`, `check_torch`).

### `BaseRandomManager.seed`

```python
@contextmanager
def seed(self, seed: int) -> Generator[None]:
    validate_seed(seed)  # before the state is read: an invalid seed changes nothing
    state = self.get_rng_state()
    try:
        self.manual_seed(seed)
        yield
    finally:
        self.set_rng_state(state)
```

`seed()` validates, so a custom manager gets the seed domain even if its `manual_seed` does not
check (the built-in managers check as well, because they are public: `Manager().manual_seed(-1)`
must not behave differently from `manual_seed(-1)`).

### The managers

```python
class PythonRandomManager(InlineDisplayMixin, BaseRandomManager):
    def get_rng_state(self) -> tuple[Any, ...]:
        return random.getstate()

    def manual_seed(self, seed: int) -> None:
        validate_seed(seed)
        random.seed(seed)

    def set_rng_state(self, state: tuple[Any, ...]) -> None:
        random.setstate(state)
```

```python
class NumpyRandomManager(InlineDisplayMixin, BaseRandomManager):
    def manual_seed(self, seed: int) -> None:
        validate_seed(seed)
        if seed < 2**32:  # same numbers as before for every seed that is not folded today
            np.random.seed(seed)
        else:  # the legacy generator accepts an array of 32-bit words: no folding
            np.random.seed([seed & 0xFFFFFFFF, seed >> 32])
```

```python
class TorchRandomManager(InlineDisplayMixin, BaseRandomManager):
    def manual_seed(self, seed: int) -> None:
        validate_seed(seed)
        torch.manual_seed(seed)  # seeds the CPU, CUDA (lazily) and MPS generators
```

`get_rng_state` and `set_rng_state` of the torch manager are kept as they are (CPU, CUDA only if
initialized, MPS; the checks of the CUDA device count and of MPS on restore).

### `RandomManagerRegistry`

```python
class RandomManagerRegistry(Registry[str, BaseRandomManager], BaseRandomManager):
    def get_rng_state(self) -> dict[str, Any]:
        return {key: manager.get_rng_state() for key, manager in self.items()}

    def manual_seed(self, seed: int) -> None:
        validate_seed(seed)  # once, so a seed rejected by one backend never half-seeds the others
        for manager in self.values():
            manager.manual_seed(seed)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if state.keys() != set(self.keys()):
            msg = (
                f"The state has the keys {sorted(state)} but the registered keys are "
                f"{sorted(self.keys())}"
            )
            raise ValueError(msg)
        errors = []
        for key, manager in self.items():
            try:
                manager.set_rng_state(state[key])
            except Exception as exc:  # noqa: BLE001  restore the others first
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Failed to restore the RNG state", errors)
```

The registration API is the one of `Registry` (`register`, `register_many`, `unregister`, `copy`,
`get`, `has`, `in`, `[]`, `clear`, `equal`, `register_lazy`, `keys`, `items`): no wrapper, no
second spelling, one thread-safe implementation that every package already uses. A probe of this
class on the previous version shows that `copy()` keeps the class, `equal` compares two copies as equal, and
`seed()` works; the `repr` is the one of `Registry` (the line `(state): Registry(...)` disappears).
The class is about 35 lines instead of 272.

A registry is a manager, so a registry can be registered in another registry (a project registry
holding the default one and its own generators). The state is then a nested dict, and the keys rule
applies at each level.

### The default registry

```python
def _register_default_managers(registry: RandomManagerRegistry) -> None:
    registry.register("random", PythonRandomManager())
    if is_numpy_available():
        from coola.random.numpy import NumpyRandomManager

        registry.register("numpy", NumpyRandomManager())
    if is_torch_available():
        from coola.random.torch import TorchRandomManager

        registry.register("torch", TorchRandomManager())
```

The function-level imports replace `import_object("...")()`: they are the idiom of the rest of the
package and are visible to type checkers. The two branches are tested by patching
`is_numpy_available` / `is_torch_available` to `False` and to `True` (unit), and by the existing
integration test with a real import (no `# pragma: no cover`, ). The first call imports the
installed libraries on purpose (a library imported after the first seed would not be seeded): the
cost is documented in the docstring of `manual_seed` and `random_seed`, and in the user guide.

### Module layout

```
coola/random/
  __init__.py     exports, lazy exports of the optional modules
  base.py         BaseRandomManager (contract + seed())
  seed.py         MAX_SEED, validate_seed
  python.py       PythonRandomManager
  numpy.py        NumpyRandomManager
  torch.py        TorchRandomManager, get_torch_generator, get_random_seed
  registry.py     RandomManagerRegistry (35 lines)
  interface.py    module functions, default registry
```

`random.py` is renamed `python.py`; everything else keeps its place, so the tests keep their
mirror (`tests/unit/random/test_python.py`).

## Decisions

| #   | Decision                                                                                                                       | Why                                                                                                                                                                                                                                                                               | Alternative rejected                                                                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `RandomManagerRegistry` subclasses `Registry[str, BaseRandomManager]` and `BaseRandomManager`.                                 | Removes 240 lines of forwarding and docstrings, gives `equal`, `copy`, `register_lazy`, `clear`, locking and fork reset for free, and keeps the composite property (a registry is a manager).                                                                                     | Composition, as before: two spellings of every operation to maintain. A mixin base class for stateless managers: one more class for three `__eq__` / `__hash__` pairs.                      |
| D2  | NumPy: `seed < 2**32` keeps `np.random.seed(seed)`; larger seeds use `np.random.seed([low, high])`.                            | Injective on the whole domain and **every seed that works before and is not folded gives the same numbers**. Only seeds `>= 2**32`, which silently collided, change. Probe: the array stream differs from the scalar one and from the other array.                                | Always an array: changes the numbers of every user. Keep folding and document: the seed domain would be a lie. Reject seeds `>= 2**32` for NumPy only: one library would decide the domain. |
| D3  | `set_rng_state` raises `ValueError` if the keys differ and restores all managers before it raises.                             | A restore that does not restore is a reproducibility bug, and a log line is not an error channel. Restoring a state from a machine with torch on a machine without torch is a user error to fix, not to log. A failure of one manager must not leave the others seeded.           | Warn and continue, as before. Strict only for missing keys: asymmetric. `ExceptionGroup` always: forces `except* ValueError` on the common single failure (one error is raised as is).      |
| D4  | Every public entry validates the seed; `validate_seed` is called by `seed()` and by each built-in manager.                     | One domain `[0, 2**63)` for all libraries is what makes `manual_seed(42)` mean one thing. A custom manager inherits the check through `seed()`. The cost is one `isinstance` and one comparison.                                                                                  | A template method (`manual_seed` concrete calling an abstract `_manual_seed`): a private abstract method is a worse extension point than the public one the users already know.             |
| D5  | The default registry imports the installed libraries when it is built, on the first use.                                       | A library imported after `manual_seed(42)` would keep its default random seed, which is the failure users cannot see. The cost (about a second for torch) is paid once and documented. A test covers the case.                                                                    | Seed only the libraries already in `sys.modules`: fast, but `manual_seed(42); import torch` silently does nothing. Import-time hooks: fragile.                                              |
| D6  | `numpy_seed` and `torch_seed` are removed.                                                                                     | `NumpyRandomManager().seed(42)` is the same context manager with the same name as every other manager, and `random_seed(42)` covers the common case. Two functions, two lazy exports and two doc sections less.                                                                   | Keep as shortcuts: they are not used, and a third spelling (`random_seed`, `Manager().seed`, `xxx_seed`) is what the user guide has to explain before.                                      |
| D7  | `get_torch_generator` and `get_random_seed` stay, are exported lazily, take keyword-only options, and the seed range is fixed. | `get_torch_generator` is the documented way to get a seeded independent generator for threads (the "Not guaranteed" part sends the reader there). `get_random_seed` returns a valid seed `[0, 2**63 - 1)` (`torch.randint` rejects `2**63`), so its output goes to `manual_seed`. | Delete both: leaves threads without a helper. Add NumPy twins: `numpy.random.default_rng(seed)` is already the one-liner, and the package manages global state.                             |
| D8  | No lock, no thread-local state, no per-thread isolation.                                                                       | The generators are global. A lock inside `random_seed` would not stop other code from drawing, and would give the false impression of safety. The limit is stated in the properties.                                                                                              | A `threading.RLock` in `seed()`: serialises seeded blocks only; deadlock-prone with a `func` that waits for another thread.                                                                 |
| D9  | `PythonRandomManager` in `python.py`; the keys of the default registry stay `"random"`, `"numpy"`, `"torch"`.                  | The class name says "python", the key says what the user types in `random.seed`. The keys appear in saved states and in the plugin documentation: they do not change.                                                                                                             | `StdlibRandomManager`: not a word the user guide uses.                                                                                                                                      |
| D10 | `random_seed(seed, *, registry=None)`, `register_managers(mapping, *, exist_ok=False)`.                                        | The rule of the repository and the name says what it is.                                                                                                                                                                                                                          | Keep positional: inconsistent with `hash_object`, `recursive_apply`.                                                                                                                        |
| D11 | `Registry.register(key, value, exist_ok)` stays positional here.                                                               | It is the primitive of `coola.registry`, shared by every package; its signature is a separate item that this package must not decide alone.                                                                                                                                       | Change it here: breaks the sibling packages in a review about `random`.                                                                                                                     |

## What is deliberately not changed

- **No `Generator` management.** Tracking `random.Random`, `default_rng` and `torch.Generator`
  objects would need weak references to every generator and a policy for the ones the package does
  not know. It is a different package; the helpers `get_torch_generator` and `default_rng(seed)`
  are the answer.
- **No automatic seeding of other libraries** (JAX uses explicit keys, TensorFlow and others can
  come as plugins through `coola.random.managers`).
- **No `hash_object`-style cache or engine.** There is nothing to cache.
- **No change of the aliases** `get_default_random_manager_registry` / `get_default_registry`: it
  is the convention of the five sibling packages and its fate is a cross-package decision
  (`review/src-coola-review.md`).
- **No change of `Registry.register(key, value, exist_ok)`**.
- **The keys of the states** (`"random"`, `"numpy"`, `"torch"`, `"torch.cuda"`, `"torch.mps"`) and
  the range `[0, 2**63)`.
