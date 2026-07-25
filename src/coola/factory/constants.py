r"""Define some constants used in the package.

These constants can be used outside the package to be robust to
naming change.

Example:
```pycon

>>> from coola.factory import factory, OBJECT_TARGET
>>> config = {OBJECT_TARGET: "collections.Counter", "a": 4, "b": 2}
>>> obj = factory(**config)
>>> obj
Counter({'a': 4, 'b': 2})

```
"""

from __future__ import annotations

__all__ = ["OBJECT_INIT", "OBJECT_TARGET"]

OBJECT_INIT = "_init_"
OBJECT_TARGET = "_target_"
