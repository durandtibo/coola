from __future__ import annotations

import re
import time

from coola.identifier import ObjectIdGenerator, generate_object_id

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


###################################
#     Tests for ObjectIdGenerator     #
###################################


def test_object_id_generator_generate_returns_str() -> None:
    assert isinstance(ObjectIdGenerator().generate(), str)


def test_object_id_generator_generate_length() -> None:
    assert len(ObjectIdGenerator().generate()) == 24


def test_object_id_generator_generate_matches_pattern() -> None:
    assert OBJECT_ID_PATTERN.match(ObjectIdGenerator().generate())


def test_object_id_generator_generate_is_unique() -> None:
    generator = ObjectIdGenerator()
    assert generator.generate() != generator.generate()


def test_object_id_generator_instances_use_independent_process_values() -> None:
    first = ObjectIdGenerator().generate()
    second = ObjectIdGenerator().generate()
    # The process-value segment (bytes 4-9, i.e. hex chars 8-18) is
    # independently randomized per instance, so it almost certainly
    # differs between two freshly created generators.
    assert first[8:18] != second[8:18]


def test_object_id_generator_generate_counter_wraps() -> None:
    generator = ObjectIdGenerator()
    generator._counter = (1 << 24) - 2
    first = generator.generate()
    second = generator.generate()
    assert int(first[18:], 16) == (1 << 24) - 1
    assert int(second[18:], 16) == 0


def test_generate_object_id_uses_shared_default_generator() -> None:
    from coola.identifier import objectid

    assert generate_object_id()[8:18] == objectid._default_generator._process_value.hex()
