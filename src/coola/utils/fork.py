r"""Reset the state of the objects in a forked process.

A process created with ``os.fork`` (e.g. a worker of a pre-fork server,
a ``torch.utils.data.DataLoader`` worker or ``multiprocessing`` with
the ``fork`` start method) starts with a copy of the memory of its
parent, but only with the thread that forked. Two kinds of state need a
reset in the child process:

- the state of the identifier generators: without a reset, the parent
  and its children would produce the same random values and the same
  counters, so they could produce the same identifiers;
- the locks: a lock that another thread of the parent held at the time
  of the fork stays locked forever in the child, because the owner does
  not exist there.

The objects that register with ``track_fork`` are reset in the child
process right after the fork.
"""

from __future__ import annotations

__all__ = ["ForkAware", "track_fork"]

import os
import weakref
from typing import Protocol


class ForkAware(Protocol):
    r"""Protocol for the objects that need to reset their state in a
    forked child process."""

    def _reset_after_fork(self) -> None:
        r"""Reset the state in a forked child process."""
        ...  # pragma: no cover


_TRACKED: weakref.WeakSet[ForkAware] = weakref.WeakSet()


def track_fork(obj: ForkAware) -> None:
    r"""Reset an object in the child process of every future fork.

    The object is only weakly referenced: it is not kept alive by this
    function.

    Args:
        obj: The object to track. Its ``_reset_after_fork`` method is
            called in the child process after a fork.

    Example:
        ```pycon
        >>> from coola.utils.fork import track_fork
        >>> class Counter:
        ...     def __init__(self):
        ...         self.value = 0
        ...         track_fork(self)
        ...     def _reset_after_fork(self):
        ...         self.value = 0
        ...
        >>> counter = Counter()

        ```
    """
    _TRACKED.add(obj)


def _after_fork_in_child() -> None:
    r"""Reset the tracked objects in the child process."""
    for obj in list(_TRACKED):
        obj._reset_after_fork()


# ``os.register_at_fork`` is not available on every platform (e.g. Windows).
if hasattr(os, "register_at_fork"):  # pragma: no branch
    os.register_at_fork(after_in_child=_after_fork_in_child)
