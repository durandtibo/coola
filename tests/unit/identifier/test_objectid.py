from __future__ import annotations

import re
import time

from coola.identifier import generate_object_id

OBJECT_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")


def test_generate_object_id_returns_str() -> None:
    assert isinstance(generate_object_id(), str)


def test_generate_object_id_length() -> None:
    assert len(generate_object_id()) == 24


def test_generate_object_id_matches_pattern() -> None:
    assert OBJECT_ID_PATTERN.match(generate_object_id())


def test_generate_object_id_is_unique() -> None:
    assert generate_object_id() != generate_object_id()


def test_generate_object_id_many_calls_are_unique() -> None:
    values = {generate_object_id() for _ in range(1000)}
    assert len(values) == 1000


def test_generate_object_id_sorts_by_creation_time() -> None:
    first = generate_object_id()
    time.sleep(1.01)
    second = generate_object_id()
    assert first < second


def test_generate_object_id_encodes_current_timestamp() -> None:
    before = int(time.time())
    object_id = generate_object_id()
    after = int(time.time())
    timestamp = int(object_id[:8], 16)
    assert before <= timestamp <= after
