from __future__ import annotations

import datetime
from collections import UserString
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from uuid import UUID

from coola.jsonable import ConvertTransformer
from coola.jsonable.stdlib import get_default_transformers


def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [
        UserString,
        PurePath,
        UUID,
        Decimal,
        bytes,
        bytearray,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
        Enum,
    ]


def test_get_default_transformers_are_conversions() -> None:
    assert all(isinstance(t, ConvertTransformer) for t in get_default_transformers().values())
