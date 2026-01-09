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
different libraries in a single call.

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

### Using a Random Seed Context

Use the `random_seed` context manager to temporarily set a seed and automatically restore the
previous RNG state afterward:

```pycon

>>> import numpy
>>> from coola.random import random_seed
>>> with random_seed(42):
...     print(numpy.random.randn(2, 4))
...
[[...]]
>>> with random_seed(42):
...     print(numpy.random.randn(2, 4))
...
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

### Python's Random Module

Set the seed only for Python's `random` module:

```pycon

>>> import random
>>> from coola.random.random import RandomRandomManager
>>> manager = RandomRandomManager()
>>> manager.manual_seed(42)
>>> random.randint(1, 100)
81

```

### NumPy

Set the seed only for NumPy's random module:

```pycon

>>> import numpy as np
>>> from coola.random import numpy_seed
>>> with numpy_seed(42):
...     np.random.rand(3)
...
array([...])

```

Or use the NumPy random manager directly:

```pycon

>>> import numpy as np
>>> from coola.random.numpy import NumpyRandomManager
>>> manager = NumpyRandomManager()
>>> manager.manual_seed(42)
>>> np.random.rand(3)
array([...])

```

### PyTorch

Set the seed only for PyTorch's random module:

```pycon

>>> import torch
>>> from coola.random import torch_seed
>>> with torch_seed(42):
...     torch.rand(3)
...
tensor([...])

```

Or use the PyTorch random manager directly:

```pycon

>>> import torch
>>> from coola.random.torch import TorchRandomManager
>>> manager = TorchRandomManager()
>>> manager.manual_seed(42)
>>> torch.rand(3)
tensor([...])

```

## Advanced Usage

### Working with the Registry

The default registry manages all available random managers:

```pycon

>>> from coola.random import get_default_registry
>>> registry = get_default_registry()
>>> registry
RandomManagerRegistry(
  (state): Registry(
      (random): RandomRandomManager()
      (numpy): NumpyRandomManager()
      (torch): TorchRandomManager()
    )
)

```

### Registering Custom Managers

You can register custom random managers to support additional libraries:

```pycon

>>> from coola.random import register_managers, RandomRandomManager
>>> # Register a custom manager
>>> register_managers({"custom": RandomRandomManager()})  # doctest: +SKIP

```

To create a custom manager, extend `BaseRandomManager`:

```pycon

>>> from coola.random import BaseRandomManager
>>> class MyRandomManager(BaseRandomManager):
...     def get_rng_state(self):
...         # Return current RNG state
...         return {}
...     def set_rng_state(self, state):
...         # Restore RNG state
...         pass
...     def manual_seed(self, seed):
...         # Set seed
...         pass
...

```

### Using Custom Registry

You can create and use a custom registry:

```pycon

>>> from coola.random import RandomManagerRegistry, RandomRandomManager
>>> registry = RandomManagerRegistry()
>>> registry.register("random", RandomRandomManager())
>>> registry.manual_seed(42)

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
...
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
>>> random.random()
0.6394...
>>> np.random.rand()
0.3745...
>>> torch.rand(1).item()
0.8823...

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
...
>>> # Continue with normal operations
>>> more_data = np.random.rand(3)

```

## Available Random Managers

The `coola.random` package provides the following random managers:

- **`RandomRandomManager`**: For Python's built-in `random` module (always registered)
- **`NumpyRandomManager`**: For NumPy's random module (registered if NumPy is available)
- **`TorchRandomManager`**: For PyTorch's random module (registered if PyTorch is available)

Each manager implements the `BaseRandomManager` interface with three methods:

- `manual_seed(seed)`: Set the random seed
- `get_rng_state()`: Get the current RNG state
- `set_rng_state(state)`: Restore a previous RNG state

## Design Principles

The `coola.random` package design provides:

1. **Unified interface**: Control multiple RNG libraries with a single API
2. **Automatic detection**: Automatically registers managers for available libraries
3. **State management**: Save and restore complete RNG state across all libraries
4. **Context managers**: Temporarily set seeds without affecting global state
5. **Extensibility**: Easy to add support for new random number generators

## Best Practices

### Always Use Context Managers When Possible

Prefer `random_seed()` context manager over `manual_seed()` for localized reproducibility:

```pycon

>>> from coola.random import random_seed
>>> # Good: State is automatically restored
>>> with random_seed(42):
...     # reproducible code
...     pass
...

```

Instead of:

```pycon

>>> from coola.random import manual_seed, get_rng_state, set_rng_state
>>> # Not ideal: Manual state management
>>> state = get_rng_state()
>>> manual_seed(42)
>>> # reproducible code
>>> set_rng_state(state)

```

### Set Seeds at the Beginning

Set seeds at the beginning of your script for full reproducibility:

```pycon

>>> from coola.random import manual_seed
>>> manual_seed(42)
>>> # Rest of your code...

```

### Document Seed Usage

Always document when and why you're setting seeds:

```pycon

>>> from coola.random import manual_seed
>>> # Set seed for reproducible results in experiments
>>> manual_seed(42)

```

## See Also

- Python's [`random` module](https://docs.python.org/3/library/random.html)
- NumPy's [Random sampling](https://numpy.org/doc/stable/reference/random/index.html)
- PyTorch's [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [`coola.registry`](registry.md): For understanding the registry pattern used internally
