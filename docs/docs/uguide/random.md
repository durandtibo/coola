# Random Number Generator Management

:book: This page describes the `coola.random` package, which provides a unified interface for
managing random number generator (RNG) state across different libraries.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.random` package provides a unified interface for controlling random number generators
(RNGs) from multiple libraries:

- Python's built-in `random` module (always available)
- NumPy's random module (if NumPy is installed)
- PyTorch's random module (if PyTorch is installed)

This allows you to set seeds, get and restore RNG states, and manage randomness consistently across
different libraries in a single call. A manager knows one library, the registry knows the keys, and
the base class knows the scoped context.

## Properties

| Property                | Statement                                                                                                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| One call, all libraries | `manual_seed` seeds `random`, and NumPy and PyTorch when they are installed. A library that is not installed is skipped, not an error.                                                                                            |
| One seed domain         | A seed is an `int` in `[0, 2**63)` (`bool` is rejected). Every entry that takes a seed (`manual_seed`, `random_seed`, `Manager.manual_seed`, `Manager.seed`, `get_torch_generator`) validates it before any generator is touched. |
| Seed injective          | For `random` and NumPy, two different valid seeds give different numbers: a NumPy seed `>= 2**32` is split in two 32-bit words, not folded. PyTorch's CPU generator folds the seeds modulo `2**32` itself (see below).            |
| Global generators only  | Only `random.seed`, `numpy.random.seed` (the legacy generator) and `torch.manual_seed` (all the default generators of torch) are managed. `random.Random()`, `default_rng()` and `torch.Generator()` are not.                     |
| Restorable              | `get_rng_state` then `set_rng_state` brings every managed generator back to where it was. The state is opaque: give it back to the registry that produced it, in the same process.                                                |
| Strict restore          | `set_rng_state` raises a `ValueError`, and changes nothing, if the keys of the state differ from the registered keys. It never silently restores a part of the state.                                                             |
| Complete restore        | If several managers fail to restore, all the others are still restored, then the error is raised (an `ExceptionGroup` if there are several errors).                                                                               |
| Scoped seeding          | `random_seed` (and `Manager.seed`) restores the state when the block exits, also when the block raises, and also when the seeding itself raises.                                                                                  |
| Nestable                | Nested `random_seed` blocks are independent: the inner block does not change the stream of the outer block.                                                                                                                       |
| Extensible              | A manager for another generator is added by registering one `BaseRandomManager`, in the default registry or through the plugin group `coola.random.managers`.                                                                     |
| Lazy                    | Importing `coola.random` imports no optional library. The first use of the default registry imports the installed ones, so a library imported after the first seed is still seeded.                                               |
| Isolated registries     | A registry built by the user is independent from the default one; `copy()` gives an independent registry that shares the managers.                                                                                                |

Not guaranteed:

- **Threads.** The generators are process-global. Two threads that call `random_seed` at the same
  time interleave their saves and restores, and the state at exit is the one of whoever restores
  last; a thread that draws numbers while another seeds gets numbers of the other's stream. Use one
  `random_seed` block at a time, or independent generators (`numpy.random.default_rng(seed)`,
  `get_torch_generator(seed)`) in threads.
- **Independent generators.** `random.Random()`, `numpy.random.default_rng()` and `torch.Generator()`
  are not seeded, and their state is not saved or restored. Seed them explicitly.
- **CUDA created inside a scoped block.** The CUDA state is saved only if CUDA is already initialized
  when the state is read (reading it would initialize CUDA, which fails in a forked worker). A CUDA
  context that the block initializes is seeded by the block and not restored by it.
- **PyTorch seeds.** The CPU generator of PyTorch folds the seed modulo `2**32`: the seeds `1` and
  `2**32 + 1` give the same numbers.
- **Equality of states.** A state holds NumPy arrays and tensors: compare two states with
  `objects_are_equal`, not with `==`.
- **Reproducibility across versions.** The package seeds the generators; the numbers a library draws
  from a seed are the library's business. Kernels that are not deterministic
  (`torch.use_deterministic_algorithms`) are out of scope.

## Basic Usage

### Setting a Random Seed

Use `manual_seed()` to set the seed for all available random number generators:

```pycon
>>> from coola.random import manual_seed
>>> manual_seed(42)
>>> # Now all RNGs (random, numpy, torch) are seeded with 42

```

This sets the seed for:

- Python's `random` module
- NumPy's random generator (if NumPy is available)
- PyTorch's random generator (if PyTorch is available)

The first call creates the default registry, which imports NumPy and PyTorch when they are
installed (about a second). They are imported at that point, and not when they are used later, so
that a library that is imported after the call is still seeded. The same applies to `random_seed()`.

### Validating a Seed

`manual_seed()` validates the seed before it seeds any library. `validate_seed()` is available to
validate a seed without seeding: it accepts an `int` in the range `[0, 2**63)`, raises a
`TypeError` for another type and a `ValueError` for a seed out of range:

```pycon
>>> from coola.random import validate_seed
>>> validate_seed(42)
>>> validate_seed(-1)
Traceback (most recent call last):
    ...
ValueError: seed must be in the range [0, 2**63), got -1
>>> validate_seed("42")
Traceback (most recent call last):
    ...
TypeError: seed must be an int, got <class 'str'>

```

### Using a Random Seed Context

Use the `random_seed` context manager to temporarily set a seed and automatically restore the
previous RNG state afterward:

```pycon
>>> import numpy
>>> from coola.random import random_seed
>>> with random_seed(42):
...     print(numpy.random.randn(2, 4))
[[...]]
>>> with random_seed(42):
...     print(numpy.random.randn(2, 4))
[[...]]

```

The context manager ensures that:

1. The RNG state is saved before setting the seed
2. The specified seed is applied
3. The original RNG state is restored after the context exits

This is useful for reproducible code blocks without affecting the global RNG state.

### Getting and Setting RNG State

You can manually save and restore the entire RNG state:

```pycon
>>> from coola.random import get_rng_state, set_rng_state
>>> # Save current state
>>> state = get_rng_state()
>>> # ... do some random operations ...
>>> # Restore previous state
>>> set_rng_state(state)

```

The state is a dictionary containing the state of all registered random managers:

```pycon
>>> from coola.random import get_rng_state
>>> state = get_rng_state()
>>> state.keys()
dict_keys(['random', 'numpy', 'torch'])

```

## Library-Specific Seeds

Each manager has a `seed` context manager, which seeds one library and restores its state when the
block exits.

### Python's Random Module

```pycon
>>> import random
>>> from coola.random import PythonRandomManager
>>> with PythonRandomManager().seed(42):
...     first = random.random()
>>> with PythonRandomManager().seed(42):
...     second = random.random()
>>> first == second
True

```

### NumPy

```pycon
>>> import numpy as np
>>> from coola.random import NumpyRandomManager
>>> with NumpyRandomManager().seed(42):
...     np.random.rand(3)
array([...])

```

NumPy's legacy generator only accepts a scalar seed between `0` and `2**32 - 1`.
`NumpyRandomManager` gives a seed below `2**32` as is, and splits a larger seed in two 32-bit words,
so two different seeds in `[0, 2**63)` never give the same numbers.

### PyTorch

```pycon
>>> import torch
>>> from coola.random import TorchRandomManager
>>> with TorchRandomManager().seed(42):
...     torch.rand(3)
tensor([...])

```

`torch.manual_seed` seeds the CPU, CUDA and MPS generators, so one call is enough.

### Independent Generators

The generators created with `random.Random()`, `numpy.random.default_rng()` and `torch.Generator()`
are not managed. Seed them explicitly. `get_torch_generator` creates a seeded `torch.Generator`
(`device` is keyword-only), and `get_random_seed` derives a seed that is valid for `manual_seed`
from another seed:

```pycon
>>> import numpy as np
>>> import torch
>>> from coola.random import get_random_seed, get_torch_generator
>>> rng = np.random.default_rng(42)
>>> rng.random(3)
array([...])
>>> generator = get_torch_generator(42)
>>> torch.rand(3, generator=generator)
tensor([...])
>>> 0 <= get_random_seed(42) < 2**63 - 1
True

```

## Advanced Usage

### Working with the Registry

!!! tip "Unambiguous alias"

    `get_default_registry` exists in several `coola` subpackages with different return types.
    `get_default_random_manager_registry` is an alias of the same function with a more explicit name, which avoids
    import collisions.

The default registry manages all available random managers, in registration order:

```pycon
>>> from coola.random import get_default_registry
>>> registry = get_default_registry()
>>> registry
RandomManagerRegistry(
  (random): PythonRandomManager()
  (numpy): NumpyRandomManager()
  (torch): TorchRandomManager()
)

```

`RandomManagerRegistry` is a [`Registry`](registry.md) of managers (`register`, `register_many`,
`unregister`, `copy`, `in`, `[]`, ...) and it is itself a manager: it seeds, saves and restores all
its managers, and the state is a `dict` that maps each key to the state of its manager.

### Using a Custom Registry

A registry built by the user is independent from the default one. `copy()` gives an independent
registry that shares the managers:

```pycon
>>> from coola.random import RandomManagerRegistry, PythonRandomManager, random_seed
>>> registry = RandomManagerRegistry()
>>> registry.register("random", PythonRandomManager())
>>> with random_seed(42, registry=registry):
...     pass
>>> other = registry.copy()
>>> other.register("other", PythonRandomManager())
>>> "other" in registry
False

```

A registry is a manager, so a registry can be registered in another registry (for example a project
registry that holds the default one and its own generators). The state is then a nested `dict`, and
the keys rule of `set_rng_state` applies at each level:

```pycon
>>> from coola.random import RandomManagerRegistry, PythonRandomManager
>>> inner = RandomManagerRegistry({"random": PythonRandomManager()})
>>> outer = RandomManagerRegistry({"inner": inner})
>>> outer.get_rng_state().keys()
dict_keys(['inner'])
>>> outer.get_rng_state()["inner"].keys()
dict_keys(['random'])

```

### Strict Restore

`set_rng_state` requires the keys of the state to be the registered keys. A state saved with torch
cannot be restored into a registry without torch: it raises a `ValueError` and nothing is restored,
instead of silently restoring a part of the state.

```pycon
>>> from coola.random import RandomManagerRegistry, PythonRandomManager
>>> registry = RandomManagerRegistry({"random": PythonRandomManager()})
>>> registry.set_rng_state({"random": registry.get_rng_state()["random"], "torch": None})
Traceback (most recent call last):
    ...
ValueError: The state has the keys ['random', 'torch'] but the registered keys are ['random']

```

If a manager fails to restore its state, the other managers are still restored before the error is
raised. One error is raised as is; several errors are raised as an `ExceptionGroup`.

### Writing a Manager

To support another generator, extend `BaseRandomManager`. A manager has to follow this contract
(checked for the built-in managers by `tests/unit/random/test_contract.py`):

1. `get_rng_state()` returns an opaque object that does not alias the live generator.
2. `set_rng_state(get_rng_state())` leaves the draws unchanged.
3. `manual_seed(seed)` validates the seed with `validate_seed` before it touches the generator, and
   the same seed gives the same draws.
4. A manager is stateless and cheap to build: it is shared by registries, equal to another instance
   of its class and hashable (`__eq__` and `__hash__`).

`seed()` validates the seed too, so a manager gets the seed domain even if its `manual_seed` does not
check. This manager handles a module-level generator:

```pycon
>>> import random
>>> from coola.random import BaseRandomManager, RandomManagerRegistry, validate_seed
>>> RNG = random.Random()
>>> class GlobalRngManager(BaseRandomManager):
...     def __eq__(self, other):
...         return type(other) is type(self)
...
...     def __hash__(self):
...         return hash(type(self))
...
...     def get_rng_state(self):
...         return RNG.getstate()
...
...     def manual_seed(self, seed):
...         validate_seed(seed)
...         RNG.seed(seed)
...
...     def set_rng_state(self, state):
...         RNG.setstate(state)
>>> registry = RandomManagerRegistry({"global_rng": GlobalRngManager()})
>>> with registry.seed(42):
...     first = RNG.random()
>>> with registry.seed(42):
...     second = RNG.random()
>>> first == second
True

```

### Registering a Manager in the Default Registry

`register_managers` adds managers to the default registry for the whole process (`exist_ok` is
keyword-only). Use `temporary_default_registries` to scope the registration, for example in tests:

```pycon
>>> from coola.random import PythonRandomManager, get_default_registry, register_managers
>>> from coola.registry.default import temporary_default_registries
>>> with temporary_default_registries():
...     register_managers({"custom": PythonRandomManager()})
...     "custom" in get_default_registry()
True
>>> "custom" in get_default_registry()
False

```

A package can also register its managers through the plugin group `coola.random.managers` (see
[Extending coola](extending.md)). The entry point is a function that takes the default registry:

```toml
# pyproject.toml of the plugin package
[project.entry-points."coola.random.managers"]
mypkg = "mypkg.coola_plugin:register"
```

```python
# mypkg/coola_plugin.py
def register(registry):
    registry.register("mylib", MyLibRandomManager())
```

## Common Use Cases

### Reproducible Experiments

Ensure reproducibility in scientific experiments:

```pycon
>>> import numpy as np
>>> import torch
>>> from coola.random import manual_seed
>>> manual_seed(42)
>>> # All random operations are now reproducible
>>> np.random.rand(3)
array([...])
>>> torch.rand(3)
tensor([...])

```

### Reproducible Code Blocks

Make a specific code block reproducible without affecting the global state:

```pycon
>>> import numpy as np
>>> from coola.random import random_seed
>>> # Generate some random data
>>> data1 = np.random.rand(5)
>>> # This block is reproducible
>>> with random_seed(123):
...     reproducible_data = np.random.rand(5)
>>> # Continue with original RNG state
>>> data2 = np.random.rand(5)

```

Internally, the context manager saves and restores RNG state before and after the block.
It is equivalent to this code.

```pycon
>>> from coola.random import get_rng_state, set_rng_state, manual_seed
>>> # Save initial state
>>> initial_state = get_rng_state()
>>> # Run test with specific seed
>>> manual_seed(42)
>>> # ... make code reproducible. ...
>>> # Restore initial state
>>> set_rng_state(initial_state)

```

### Cross-Library Seeding

Seed all libraries at once for consistent results:

```pycon
>>> import random
>>> import numpy as np
>>> import torch
>>> from coola.random import manual_seed
>>> # Seed all libraries with one call
>>> manual_seed(42)
>>> # All libraries use the same seed
>>> val1 = random.random()
>>> val2 = np.random.rand()
>>> val3 = torch.rand(1).item()

```

### Temporary Seed Changes

Temporarily change seed for a specific operation:

```pycon
>>> import numpy as np
>>> from coola.random import random_seed
>>> # Normal operations
>>> normal_data = np.random.rand(3)
>>> # Use specific seed for initialization
>>> with random_seed(999):
...     initialization = np.random.randn(10, 10)
>>> # Continue with normal operations
>>> more_data = np.random.rand(3)

```

## Available Random Managers

The `coola.random` package provides the following random managers:

- **`PythonRandomManager`**: For Python's built-in `random` module (always registered)
- **`NumpyRandomManager`**: For NumPy's random module (registered if NumPy is available)
- **`TorchRandomManager`**: For PyTorch's random module (registered if PyTorch is available)

Each manager implements the `BaseRandomManager` interface with three methods:

- `manual_seed(seed)`: Set the random seed
- `get_rng_state()`: Get the current RNG state
- `set_rng_state(state)`: Restore a previous RNG state

## Threads and Processes

The generators are process-global and `coola.random` does not lock them: it cannot, because other
code draws from the same generators. In threads, use independent generators
(`numpy.random.default_rng(seed)`, `get_torch_generator(seed)`) instead of `random_seed`.

The CUDA state is read only if CUDA is already initialized, so `get_rng_state` can be called in a
forked worker. A child process restores a state with `set_rng_state`.

## Best Practices

### Prefer `random_seed`

Prefer the `random_seed()` context manager over `manual_seed()` for localized reproducibility: the
state is restored when the block exits.

```pycon
>>> from coola.random import random_seed
>>> with random_seed(42):
...     pass  # reproducible code

```

### Set Seeds at the Beginning

Set seeds at the beginning of your script for full reproducibility:

```pycon
>>> from coola.random import manual_seed
>>> manual_seed(42)

```

### Use a Generator in Threads

Give each thread its own generator, seeded explicitly, instead of seeding the global ones.

## See Also

- Python's [`random` module](https://docs.python.org/3/library/random.html)
- NumPy's [Random sampling](https://numpy.org/doc/stable/reference/random/index.html)
- PyTorch's [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [`coola.registry`](registry.md): For understanding the registry pattern used internally
