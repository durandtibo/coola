# `coola.random`: target state and the changes to reach it

This document describes the design that `coola.random` should have, then the changes to make to the
current implementation (`main`, `3e0fce4a`) to get there. The design follows the rules of the user
guide (`docs/docs/uguide/random.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: registry, lazy imports, strict contracts, package layering where
`random` is a standalone helper), and it stays aligned with the sibling packages (`Registry`
primitives of `coola.registry`, `LazySingleton` default registries, plugin groups, keyword-only
options). Its aim is a small package that is easy to maintain: **one seed domain, one registry, one
restore rule, one manager per library**. Breaking changes are accepted, and the future state is more
important than the past one.

Evidence comes from probes run on `main` (Python 3.14, NumPy 2.5.4, torch 2.14.1 with MPS, scripts
kept outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from
the code. Nothing in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 1,044 lines of source (1,605 of unit tests) to
   call three seeding functions. About 250 of these lines are wrappers that re-implement
   `coola.registry.Registry` (`RandomManagerRegistry`) and docstrings of two functions that only
   rename `Manager().seed()`. The target removes code that exists only because of the shape of the
   current API, instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no alias,
   no deprecation period. The release notes list the changes. The one exception is deliberate and
   free: the random numbers produced by a seed that is already accepted today must not change
   (reproducibility of the user's experiments is the whole point of the package, see D2).
3. **Standalone helper.** `coola.random` imports only core packages (`registry`, `utils`,
   `display`). It is not a type-dispatch registry and does not become one (the key of a manager is a
   name, not a type).
4. **Optional libraries stay lazy.** `import coola.random` imports neither NumPy nor torch. The
   managers of the optional libraries are in their own modules, and the default registry imports
   them when it is built (D5).
5. **Process-global by nature.** The generators it manages are process-global. The package does not
   try to make them thread-local or to lock them: it cannot, because other code draws from the
   same generators. It states this limit instead (section 1, "Not guaranteed").

## 1. Properties of the sub-package

These are the rules of the user guide, completed with the rules that the code already follows
without saying it, and the ones that are missing today. Every design choice below exists to make
one of them true **by construction** instead of by the discipline of each manager.

| Property                | Statement                                                                                                                                                                                                                                              | Enforced by                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| One call, all libraries | `manual_seed` seeds `random`, and NumPy and PyTorch when they are installed. A library that is not installed is skipped, not an error.                                                                                                                 | the default registry                       |
| One seed domain         | A seed is an `int` in `[0, 2**63)` (`bool` is rejected). Every public entry that takes a seed (`manual_seed`, `random_seed`, `Manager.manual_seed`, `Manager.seed`, `Registry.manual_seed`) validates it identically, before any generator is touched. | `validate_seed`, called by every entry     |
| Seed injective          | Two different valid seeds give two different generator states for every managed library. No seed is silently folded onto another one.                                                                                                                  | the managers                               |
| Global generators only  | Only `random.seed`, `numpy.random.seed` (the legacy generator) and `torch.manual_seed` (all the default generators of torch) are managed. `random.Random()`, `default_rng()` and `torch.Generator()` are not.                                          | the managers (documented)                  |
| Restorable              | `get_rng_state` then `set_rng_state` brings every managed generator back to where it was. The state is opaque: it is given back to the registry that produced it, in the same process.                                                                 | the managers, the registry                 |
| Strict restore          | `set_rng_state` raises `ValueError`, and changes nothing, if the keys of the state differ from the registered keys. It never silently restores a part of the state.                                                                                    | `RandomManagerRegistry.set_rng_state`      |
| Complete restore        | If several managers fail to restore, all the others are still restored, then the error is raised. A failure of one library does not leave the others in the seeded state.                                                                              | `RandomManagerRegistry.set_rng_state`      |
| Scoped seeding          | `random_seed` (and `Manager.seed`) restores the state when the block exits, also when the block raises, and also when the seeding itself raises.                                                                                                       | `BaseRandomManager.seed`                   |
| Nestable                | Nested `random_seed` blocks are independent: the inner block does not change the stream of the outer block.                                                                                                                                            | LIFO save and restore                      |
| Extensible              | A manager for another generator is added by registering one `BaseRandomManager` (in the default registry, or through the plugin group `coola.random.managers`), without changing another manager.                                                      | `Registry`, the plugin group               |
| Lazy                    | Importing `coola.random` imports no optional library. The first use of the default registry imports the installed ones (so a library imported after the first seed is still seeded).                                                                   | lazy exports, `_register_default_managers` |
| Isolated registries     | A registry built by the user is independent from the default one; `copy()` gives an independent registry that shares the (stateless) managers.                                                                                                         | `Registry.copy`                            |

Not guaranteed, and documented as such:

- **Threads.** The generators are process-global. Two threads that call `random_seed` at the same
  time interleave their saves and restores, and the state at exit is the one of whoever restores
  last; a thread that draws numbers while another seeds gets numbers of the other's stream. Use one
  `random_seed` block at a time, or independent generators (`numpy.random.default_rng(seed)`,
  `get_torch_generator(seed)`) in threads.
- **Independent generators** (see "Global generators only").
- **CUDA created inside a scoped block.** The CUDA state is saved only if CUDA is already
  initialized when the state is read (reading it would initialize CUDA, which fails in a forked
  worker). A CUDA context that the block initializes is seeded by the block and not restored by it.
- **Equality of states.** A state holds NumPy arrays and tensors: compare two states with
  `objects_are_equal`, not with `==`.
- **Reproducibility across versions.** The package seeds the generators; the numbers a library
  draws from a seed are the library's business. Kernels that are not deterministic
  (`torch.use_deterministic_algorithms`) are out of scope.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Status         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The seed domain is enforced by the registry only.** `validate_seed` is called in `RandomManagerRegistry.manual_seed`, not by the managers nor by `BaseRandomManager.seed`. Calling `manual_seed` / `seed` / `numpy_seed` / `torch_seed` with `-1`, `2**64`, `"a"`, `1.5` or `True` gives, per manager: `random` accepts all five; NumPy folds `-1` and `2**64` and raises `TypeError` for `"a"` and `1.5`; torch accepts `-1`, `1.5` and `True`, raises `ValueError` for `"a"` and `2**64`. The same call has four behaviours, and the user guide states only one.      | **[verified]** |
| F2  | **The restore of the registry is neither strict nor complete.** `set_rng_state` only logs a warning for a missing key (that manager keeps its current state) and for an unregistered key (the value is dropped): the run goes on with a generator that was not restored, and the only trace is a log line. It restores in registration order and stops at the first manager that raises, so the later managers (for example `random`, registered after a failing one) are left in the seeded state. Inside `seed()`, that failure also hides the real error of the block. | **[verified]** |
| F3  | **NumPy seeds are folded.** `NumpyRandomManager.manual_seed` uses `seed % 2**32`: `manual_seed(1)` and `manual_seed(2**32 + 1)` give the same NumPy numbers while `random` and torch differ. A valid seed in `[2**32, 2**63)` is therefore not what it says. NumPy accepts an array of 32-bit words as a seed, which gives a distinct stream for every 63-bit seed.                                                                                                                                                                                                       | **[verified]** |
| F4  | **`TorchRandomManager.manual_seed` repeats what `torch.manual_seed` does.** The torch documentation states that it "sets the seed for generating random numbers on all devices": CUDA (`manual_seed_all`, lazily when CUDA is not initialized) and MPS are seeded by the same call. The extra `torch.cuda.manual_seed_all` and `torch.mps.manual_seed` calls are redundant, and `cast("Any", torch)` exists only to call the function.                                                                                                                                    | **[reading]**  |
| F5  | **`RandomManagerRegistry` re-implements `Registry`.** It holds a `Registry[str, BaseRandomManager]` and forwards `register`, `register_many`, `unregister`, `copy`, `has_manager` (= `in`) and `find_manager` (= `[]`) with 150 lines of docstrings, and adds a second spelling of each operation. Two equal registries are not equal (`==` is identity: **[verified]** `False`), while the managers define `__eq__`. The note of `review/src-coola-review.md` (section 2) already asks for this.                                                                         | **[verified]** |
| F6  | **Two public functions rename a method.** `numpy_seed(seed)` and `torch_seed(seed)` are `NumpyRandomManager().seed(seed)` and `TorchRandomManager().seed(seed)`. They need two lazy exports, two entries in the user guide and two test classes, and no other package uses them.                                                                                                                                                                                                                                                                                          | **[reading]**  |
| F7  | **Naming.** `RandomRandomManager` stutters, and its module `coola/random/random.py` is `coola.random.random`, a name that reads like the standard library. The user guide imports it as `from coola.random.random import RandomRandomManager`.                                                                                                                                                                                                                                                                                                                            | **[reading]**  |
| F8  | **`get_random_seed` and `get_torch_generator` are half-public.** They are in `torch.py` (`__all__` of that module), not exported by `coola.random`, not in the user guide and not used by `src`. `get_random_seed` returns a value in `[-2**63, 2**63 - 1)`, which `validate_seed` rejects when it is negative (about half of the values), so its output cannot be given to `manual_seed`. `get_torch_generator(random_seed, device)` has a positional `device` and the parameter is named `random_seed`, the name of the context manager of this package.                | **[reading]**  |
| F9  | **Options are positional.** `random_seed(seed, manager=None)`, `register_managers(mapping, exist_ok=False)` and `Registry.register(key, manager, exist_ok=False)` accept the option by position, against the rule "new options are keyword-only". The argument `manager` of `random_seed` is a registry, not a manager.                                                                                                                                                                                                                                                   | **[reading]**  |
| F10 | **Register hides the cost.** `register_managers` mutates the default registry for the whole process and its doctest is skipped (`# doctest: +SKIP`), so the documented way of extending the package is the only one that is not run. There is no scoped variant: tests that register a manager must use `temporary_default_registries` of `coola.utils.singleton`, which the user guide does not mention.                                                                                                                                                                 | **[reading]**  |
| F11 | **The user guide is out of line with the code and the template.** It lists "Design Principles" and "Properties" separately (the same content twice), has no "Not guaranteed" part (threads, CUDA created in a block, independent generators other than the warning box), shows a custom manager that cannot be used (no `__eq__`, a `get_rng_state` that returns `{}`, never registered or run), and teaches `from coola.random.numpy import NumpyRandomManager` next to `from coola.random import numpy_seed`.                                                           | **[reading]**  |
| F12 | **The contract of a manager is implicit.** `BaseRandomManager` says that three methods must be implemented, not what they promise: that `set_rng_state(get_rng_state())` is a no-op, that the state is opaque and independent from the live generator (no aliasing), that `manual_seed` validates its seed, that a manager is stateless and can be shared by registries. Nothing tests the three managers against the same contract.                                                                                                                                      | **[reading]**  |
| F13 | **`# pragma: no cover` on the default registration.** `_register_default_managers` hides the two lines that build the NumPy and torch managers because they depend on the environment, so the function that decides what `manual_seed` seeds has no test in the environments where a library is missing.                                                                                                                                                                                                                                                                  | **[reading]**  |

What is already right and must not regress:

- the registry design (`get_default_registry` as a `LazySingleton` with a plugin group, the
  fork-safe state of `Registry`);
- the first seed imports the installed libraries so that a library imported later is seeded
  (`test_manual_seed_seeds_numpy_even_if_it_is_imported_after`);
- `validate_seed` once in the registry before any generator is touched (a seed rejected by one
  backend does not leave the others half-seeded);
- the CUDA state is read only when CUDA is initialized (fork safety, #2259);
- the state of a manager and the state of a registry have the same shape, so a registry can be a
  manager (composition) and a registry can contain a registry;
- lazy exports of `NumpyRandomManager` and `TorchRandomManager`.

## 3. Architecture

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

### 3.1 Public API

| Name                                            | Role                                                                                                 | Change                                         |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `manual_seed(seed)`                             | Seed every manager of the default registry.                                                          | unchanged                                      |
| `random_seed(seed, *, registry=None)`           | Scoped seed; restores the state of the registry.                                                     | `manager` is now the keyword-only `registry`   |
| `get_rng_state()` / `set_rng_state(state)`      | Save / restore the state of the default registry.                                                    | `set_rng_state` is strict (D3)                 |
| `register_managers(mapping, *, exist_ok=False)` | Add managers to the default registry.                                                                | `exist_ok` keyword-only                        |
| `get_default_registry()` and its alias          | The default `RandomManagerRegistry`.                                                                 | unchanged                                      |
| `validate_seed(seed)`                           | Raise `TypeError` / `ValueError` for a seed out of the domain.                                       | unchanged                                      |
| `BaseRandomManager`                             | The extension point: three abstract methods and the concrete `seed()`.                               | `seed()` validates; documented contract        |
| `PythonRandomManager`                           | `random`.                                                                                            | renamed from `RandomRandomManager`             |
| `NumpyRandomManager`, `TorchRandomManager`      | NumPy, torch (lazy exports).                                                                         | validate; NumPy no longer folds; torch simpler |
| `RandomManagerRegistry`                         | A registry of managers that is a manager.                                                            | subclass of `Registry` (D1)                    |
| `get_torch_generator(seed, *, device="cpu")`    | A `torch.Generator` seeded with `seed` (lazy export).                                                | exported; `seed`, keyword-only `device`        |
| `get_random_seed(seed)`                         | A deterministic seed in `[0, 2**63 - 1)` derived from `seed`, valid for `manual_seed` (lazy export). | range fixed, exported                          |

Removed: `numpy_seed`, `torch_seed`, `RandomRandomManager`, `RandomManagerRegistry.has_manager`,
`find_manager`, and the warnings of `set_rng_state`.

### 3.2 The contract of a manager

Documented in `BaseRandomManager`, and checked for every manager by one parametrised unit test
(a `MANAGERS` list built with the `*_available` fixtures, like `NATIVE_SAMPLES` in `equality`):

1. `get_rng_state()` returns an opaque object that does not alias the live generator (drawing a
   number afterwards does not change it).
2. `set_rng_state(get_rng_state())` leaves the draws unchanged.
3. `manual_seed(seed)` validates the seed with `validate_seed` before it touches a generator, and
   the same seed gives the same draws.
4. Two different valid seeds give different draws (injective, F3).
5. A manager is stateless and cheap to build: it is created once, shared by registries, equal to
   another instance of its class, and hashable (`__eq__` and `__hash__` on the type).
6. Building a manager of an uninstalled library raises `ImportError` (`check_numpy`, `check_torch`).

### 3.3 `BaseRandomManager.seed`

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

### 3.4 The managers

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
        if seed < 2**32:  # same numbers as today for every seed that is not folded today
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

### 3.5 `RandomManagerRegistry`

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
class on `main` shows that `copy()` keeps the class, `equal` compares two copies as equal, and
`seed()` works; the `repr` is the one of `Registry` (the line `(state): Registry(...)` disappears).
The class is about 35 lines instead of 272.

A registry is a manager, so a registry can be registered in another registry (a project registry
holding the default one and its own generators). The state is then a nested dict, and the keys rule
applies at each level.

### 3.6 The default registry

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
integration test with a real import (no `# pragma: no cover`, F13). The first call imports the
installed libraries on purpose (a library imported after the first seed would not be seeded): the
cost is documented in the docstring of `manual_seed` and `random_seed`, and in the user guide.

### 3.7 Module layout

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

`random.py` is renamed `python.py` (F7); everything else keeps its place, so the tests keep their
mirror (`tests/unit/random/test_python.py`).

## 4. Decisions

| #   | Decision                                                                                                                       | Why                                                                                                                                                                                                                                                                               | Alternative rejected                                                                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | `RandomManagerRegistry` subclasses `Registry[str, BaseRandomManager]` and `BaseRandomManager`.                                 | Removes 240 lines of forwarding and docstrings (F5), gives `equal`, `copy`, `register_lazy`, `clear`, locking and fork reset for free, and keeps the composite property (a registry is a manager).                                                                                | Composition, as today: two spellings of every operation to maintain. A mixin base class for stateless managers: one more class for three `__eq__` / `__hash__` pairs.                       |
| D2  | NumPy: `seed < 2**32` keeps `np.random.seed(seed)`; larger seeds use `np.random.seed([low, high])`.                            | Injective on the whole domain (F3) and **every seed that works today and is not folded gives the same numbers**. Only seeds `>= 2**32`, which silently collided, change. Probe: the array stream differs from the scalar one and from the other array.                            | Always an array: changes the numbers of every user. Keep folding and document: the seed domain would be a lie. Reject seeds `>= 2**32` for NumPy only: one library would decide the domain. |
| D3  | `set_rng_state` raises `ValueError` if the keys differ and restores all managers before it raises (F2).                        | A restore that does not restore is a reproducibility bug, and a log line is not an error channel. Restoring a state from a machine with torch on a machine without torch is a user error to fix, not to log. A failure of one manager must not leave the others seeded.           | Warn and continue, as today. Strict only for missing keys: asymmetric. `ExceptionGroup` always: forces `except* ValueError` on the common single failure (one error is raised as is).       |
| D4  | Every public entry validates the seed (F1); `validate_seed` is called by `seed()` and by each built-in manager.                | One domain `[0, 2**63)` for all libraries is what makes `manual_seed(42)` mean one thing. A custom manager inherits the check through `seed()`. The cost is one `isinstance` and one comparison.                                                                                  | A template method (`manual_seed` concrete calling an abstract `_manual_seed`): a private abstract method is a worse extension point than the public one the users already know.             |
| D5  | The default registry imports the installed libraries when it is built, on the first use.                                       | A library imported after `manual_seed(42)` would keep its default random seed, which is the failure users cannot see. The cost (about a second for torch) is paid once and documented. A test covers the case.                                                                    | Seed only the libraries already in `sys.modules`: fast, but `manual_seed(42); import torch` silently does nothing. Import-time hooks: fragile.                                              |
| D6  | `numpy_seed` and `torch_seed` are removed.                                                                                     | `NumpyRandomManager().seed(42)` is the same context manager with the same name as every other manager, and `random_seed(42)` covers the common case. Two functions, two lazy exports and two doc sections less (F6).                                                              | Keep as shortcuts: they are not used, and a third spelling (`random_seed`, `Manager().seed`, `xxx_seed`) is what the user guide has to explain today.                                       |
| D7  | `get_torch_generator` and `get_random_seed` stay, are exported lazily, take keyword-only options, and the seed range is fixed. | `get_torch_generator` is the documented way to get a seeded independent generator for threads (the "Not guaranteed" part sends the reader there). `get_random_seed` returns a valid seed `[0, 2**63 - 1)` (`torch.randint` rejects `2**63`), so its output goes to `manual_seed`. | Delete both: leaves threads without a helper. Add NumPy twins: `numpy.random.default_rng(seed)` is already the one-liner, and the package manages global state.                             |
| D8  | No lock, no thread-local state, no per-thread isolation.                                                                       | The generators are global. A lock inside `random_seed` would not stop other code from drawing, and would give the false impression of safety. The limit is stated in the properties.                                                                                              | A `threading.RLock` in `seed()`: serialises seeded blocks only; deadlock-prone with a `func` that waits for another thread.                                                                 |
| D9  | `PythonRandomManager` in `python.py`; the keys of the default registry stay `"random"`, `"numpy"`, `"torch"`.                  | The class name says "python", the key says what the user types in `random.seed`. The keys appear in saved states and in the plugin documentation: they do not change.                                                                                                             | `StdlibRandomManager`: not a word the user guide uses.                                                                                                                                      |
| D10 | `random_seed(seed, *, registry=None)`, `register_managers(mapping, *, exist_ok=False)`.                                        | The rule of the repository (F9) and the name says what it is.                                                                                                                                                                                                                     | Keep positional: inconsistent with `hash_object`, `recursive_apply`.                                                                                                                        |
| D11 | `Registry.register(key, value, exist_ok)` stays positional here.                                                               | It is the primitive of `coola.registry`, shared by every package; its signature is a separate item that this package must not decide alone.                                                                                                                                       | Change it here: breaks the sibling packages in a review about `random`.                                                                                                                     |

## 5. Changes to make

One item per pull request, in this order (each one leaves the tests green):

| #   | Change                                                                                                                                                                                                                                                                                                                                                                | Finding  | Breaking                                             |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------- |
| R1  | Validate the seed in `BaseRandomManager.seed` and in the three managers. Tests: `-1`, `2**63`, `"a"`, `1.5`, `True` raise the same error for every manager, and no generator changes (one parametrised test per manager).                                                                                                                                             | F1       | `-1`, `1.5`, `True`, `2**63..` now raise on managers |
| R2  | NumPy: array seed for `seed >= 2**32`. Tests: `seed` and `seed + 2**32` differ; `seed < 2**32` gives the literal values of `main` (regression test with the literal numbers); `2**63 - 1` works.                                                                                                                                                                      | F3       | numbers for seeds `>= 2**32`                         |
| R3  | Torch: drop the CUDA and MPS calls and the `cast`. Check with `.venv/bin/ty check` that `cast` is not needed; the existing tests that patch the CUDA / MPS calls become tests of `get_rng_state` / `set_rng_state` only, plus one integration test with a real `torch.manual_seed`.                                                                                   | F4       | none                                                 |
| R4  | `RandomManagerRegistry` as a subclass of `Registry`; strict and complete `set_rng_state`; remove `has_manager` / `find_manager`. Tests: keys differ (missing, extra), nothing restored on `ValueError`, all restored when one manager raises, a single error is raised as is, two errors give an `ExceptionGroup`, nested registry, `copy()` is independent, `equal`. | F2, F5   | `has_manager`, `find_manager`, warnings, `repr`      |
| R5  | Rename `RandomRandomManager` to `PythonRandomManager` (`python.py`); remove `numpy_seed` and `torch_seed`; keyword-only `registry` and `exist_ok`; export and fix `get_torch_generator` and `get_random_seed`. Update `tests/unit/test_public_api.py`, `tests/package_checks.py`, `docs/docs/dev/add_backend.md`.                                                     | F6-F9    | names, keyword-only options, `get_random_seed` range |
| R6  | Default registration with function-level imports and no `pragma: no cover`; tests for the four combinations (neither, NumPy only, torch only, both) by patching the availability functions; an integration test with the real libraries.                                                                                                                              | F13      | none                                                 |
| R7  | The contract test over the three managers (section 3.2): no aliasing, restore is a no-op, determinism, injectivity, equality and hash, `ImportError` without the library.                                                                                                                                                                                             | F12      | none                                                 |
| R8  | Documentation: rewrite `docs/docs/uguide/random.md` (below), the docstring of `BaseRandomManager` (contract), `docs/docs/refs/random.md` (add `get_torch_generator` through the package export), the migration page, `review/src-coola-review.md` (mark the `random` item done), `review/suggestions.md`.                                                             | F10, F11 | none                                                 |

R1 to R3 are independent of R4 and can go in parallel; R5 needs R4 (names), R8 goes last. Every
bug fixed (R1, R2, R4) gets a regression test that fails without the fix (`git stash push src`).

## 6. Tests

Today: 1,605 lines of unit tests, most of them test the `Registry` wrappers (F5) and the patched
CUDA / MPS branches (F4). After R1 to R7:

- **Unit** (`tests/unit/random/`): mirrors the source tree, one behaviour per test.
  `test_seed.py` (domain, error types and messages), `test_base.py` (`seed()` validates, restores on
  an exception of the block, restores on an exception of `manual_seed`, does not seed when the seed
  is invalid), `test_python.py`, `test_numpy.py` (literal values of the first draw for `42`, no
  folding), `test_torch.py` (state keys, CUDA / MPS branches with a fake `torch.cuda` module),
  `test_registry.py` (only what the subclass adds: `get_rng_state`, `manual_seed`, `set_rng_state`,
  the rules of D3), `test_interface.py` (forwarding, `registry=` keyword, default registration of
  the four combinations).
- **Contract** (`tests/unit/random/test_contract.py`): the six rules of section 3.2 over the
  managers that are available.
- **Integration** (`tests/integration/random/`): the nested `random_seed` tests as today; every
  patched behaviour of the unit tests has a real counterpart (real `torch.manual_seed` seeds
  `torch.rand`, real NumPy array seed, real `get_rng_state` / `set_rng_state` round trip across the
  three libraries); a library imported after the first seed is seeded (exists in
  `test_lazy_imports.py`); fork (`test_fork.py`): a child process restores a state.
- **Threads** (`tests/integration/random/`, `run_threads`): two threads with `get_torch_generator`
  produce the sequences of the single-thread run (documents the supported way); no test claims that
  `random_seed` is thread-safe.
- **Property** (`tests/properties/`, exists for `random_seed`): the same numbers for the same seed,
  the state restored after the block, and now `seed != other` implies different NumPy numbers for
  `seed, other` in `[0, 2**63)`.
- Assertions on composite values use literals: `registry.get_rng_state().keys()` and the nested
  dict shape, not only single keys.

## 7. User guide (`docs/docs/uguide/random.md`)

The page keeps its examples that run as doctests and gets this shape (the template of
`uguide/recursive.md`): Overview, **Properties** (the table of section 1, once; "Design
Principles" is deleted), **Not guaranteed**, Basic usage (`manual_seed`, `random_seed`,
`get_rng_state` / `set_rng_state`, `validate_seed`), Library-specific seeds
(`NumpyRandomManager().seed(42)` etc., independent generators with `default_rng(seed)` and
`get_torch_generator`), Registry (default registry, `register_managers`, the plugin group
`coola.random.managers`, a custom registry, a registry in a registry), **Writing a manager** (a
complete, runnable manager that follows the contract of section 3.2, registered in a scoped
registry: `with temporary_default_registries():` or a user registry, so that the example runs and
leaves nothing behind), Threads and processes (the limit, the fork test), Best practices (the three
short items that remain useful: prefer `random_seed`, seed at the start, use a `Generator` for
threads). The two NumPy warning boxes merge into one sentence of the properties table ("Seed
injective").

## 8. What is deliberately not changed

- **No `Generator` management.** Tracking `random.Random`, `default_rng` and `torch.Generator`
  objects would need weak references to every generator and a policy for the ones the package does
  not know. It is a different package; the helpers `get_torch_generator` and `default_rng(seed)`
  are the answer.
- **No automatic seeding of other libraries** (JAX uses explicit keys, TensorFlow and others can
  come as plugins through `coola.random.managers`).
- **No `hash_object`-style cache or engine.** There is nothing to cache.
- **No change of the aliases** `get_default_random_manager_registry` / `get_default_registry`: it
  is the convention of the five sibling packages and its fate is a cross-package decision
  (`review/src-coola-review.md`, section 2).
- **No change of `Registry.register(key, value, exist_ok)`** (D11).
- **The keys of the states** (`"random"`, `"numpy"`, `"torch"`, `"torch.cuda"`, `"torch.mps"`) and
  the range `[0, 2**63)`.

## 9. Size of the result

| Part                       | Today (lines) | Target (estimate) | Why                                                           |
| -------------------------- | ------------- | ----------------- | ------------------------------------------------------------- |
| `registry.py`              | 272           | 60                | subclass of `Registry` (D1)                                   |
| `interface.py`             | 244           | 190               | no `manager` argument, shorter default registration docstring |
| `numpy.py`                 | 96            | 75                | `numpy_seed` removed, array seed added                        |
| `torch.py`                 | 179           | 150               | `torch_seed` removed, `manual_seed` shortened                 |
| `random.py` / `python.py`  | 47            | 45                | renamed, one validation line                                  |
| `base.py`, `seed.py`, init | 206           | 215               | contract in the docstring of `BaseRandomManager`              |
| **Source**                 | **1,044**     | **about 735**     |                                                               |
| Unit tests                 | 1,605         | about 1,100       | the wrappers are no longer tested; the contract test is added |

## 10. Order of risk

1. **R2 (NumPy numbers).** Only seeds `>= 2**32` change; the regression test pins the literal first
   draws of seeds `0`, `42` and `2**32 - 1` taken on `main` before the change.
2. **R4 (`set_rng_state` strict).** A caller that saved a state with torch and restores it without
   torch now gets a `ValueError` instead of a log line. This is the intended change; the migration
   page shows the one-line fix (restore into a registry with the same keys).
3. **R1 (validation in the managers).** Code that called `NumpyRandomManager().manual_seed(-1)` or
   `torch_seed(1.5)` now raises. It was relying on a folded or accepted invalid seed.
4. **R5 (renames).** Mechanical; a search for `RandomRandomManager`, `numpy_seed`, `torch_seed`
   finds all the uses (none in `src` outside the package).
