from __future__ import annotations

import re

from coola.identifier import generate_uuid4

UUID4_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_generate_uuid4_returns_str() -> None:
    assert isinstance(generate_uuid4(), str)


def test_generate_uuid4_length() -> None:
    assert len(generate_uuid4()) == 36


def test_generate_uuid4_matches_pattern() -> None:
    assert UUID4_PATTERN.match(generate_uuid4())


def test_generate_uuid4_is_unique() -> None:
    assert generate_uuid4() != generate_uuid4()


def test_generate_uuid4_many_calls_are_unique() -> None:
    values = {generate_uuid4() for _ in range(1000)}
    assert len(values) == 1000
