r"""Contain functions that turn values into text for messages, ``repr``
and reports.

The package exports nothing: import a function from its module.

- ``coola.formatting.structure``: ``str_*`` and ``repr_*`` of sequences and
  mappings, and indentation.
- ``coola.formatting.quantity``: human-readable byte sizes and durations,
  and ``make_bar``.
- ``coola.formatting.text``: ``count_lines``, ``remove_empty_lines``,
  ``slugify``, ``truncate_str`` and ``char_diff_summary``.
- ``coola.formatting.secret``: ``mask_secret``.
- ``coola.formatting.diff``: ``unified_diff``.

The functions are pure: the same input gives the same string.
"""

from __future__ import annotations

__all__: list[str] = []
