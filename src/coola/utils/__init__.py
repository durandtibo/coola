r"""Shared helper utilities used by public and internal ``coola``
APIs.

The package exports nothing: import a function from its module. The core
helpers, used by the other ``coola`` packages, are ``imports`` (optional
dependencies), ``fallback``, ``fork``, ``integer``, ``introspection``,
``mapping``, ``path``, ``safe``, ``selection``, ``singleton`` (e.g.
``LazySingleton``), ``stats`` and ``version``. The standalone helpers are
``array``, ``tensor``, ``batching``, ``bloom_filter``, ``date``,
``env_vars``, ``file_size``, ``git``, ``logging``, ``lru``, ``password``,
``recursion`` and ``timing``. Functions that turn values into text are in
``coola.formatting``.
"""

from __future__ import annotations

__all__: list[str] = []
