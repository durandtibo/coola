r"""Define the transformers of the types of the standard library."""

from __future__ import annotations

__all__ = ["get_default_transformers"]

import base64
import datetime
from collections import UserString
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import TYPE_CHECKING, Any
from uuid import UUID

from coola.jsonable.transformers import ConvertTransformer

if TYPE_CHECKING:
    from coola.recursive.base import Transformer


def get_default_transformers() -> dict[type, Transformer[Any]]:
    r"""Get the default transformers of the standard library types.

    Returns:
        The transformers of the types and their associated types.

    Notes:
        A ``str`` enum member or an ``IntEnum`` member is a ``str`` or
        an ``int`` for the Method Resolution Order, so it is a JSON
        scalar and it is kept.

    Example:
        ```pycon
        >>> from coola.jsonable.stdlib import get_default_transformers
        >>> sorted(t.__name__ for t in get_default_transformers())[:3]
        ['Decimal', 'Enum', 'PurePath']

        ```
    """
    to_str = ConvertTransformer(str)
    to_iso = ConvertTransformer(lambda value: value.isoformat())
    to_base64 = ConvertTransformer(lambda value: base64.b64encode(value).decode("ascii"))
    return {
        # ``UserString`` is a ``Sequence`` of ``UserString``: it must not be a list of characters
        UserString: to_str,
        PurePath: to_str,
        UUID: to_str,
        Decimal: to_str,
        bytes: to_base64,
        bytearray: to_base64,
        datetime.date: to_iso,
        datetime.datetime: to_iso,
        datetime.time: to_iso,
        datetime.timedelta: ConvertTransformer(lambda value: value.total_seconds()),
        Enum: ConvertTransformer(lambda value: value.value),
    }
