r"""Provide identifiers: content-derived, time-ordered and human-
friendly.

The stable identifiers (``generate_stable_content_id``,
``generate_stable_uuid5``) are derived from nested data through
``coola.hashing``. The others are generated: time-ordered (ULID, UUIDv7,
Snowflake, ObjectId) or checksummed.

Every generator returns a ``str`` identifier except
``SnowflakeIdGenerator.generate``, which returns a plain ``int`` (a
Snowflake ID is defined as a 64-bit integer, e.g. for use as a database
``BIGINT`` primary key).
"""

from __future__ import annotations

__all__ = [
    "SnowflakeIdGenerator",
    "extract_object_id_timestamp_s",
    "extract_snowflake_timestamp_ms",
    "extract_ulid_timestamp_ms",
    "extract_uuid7_timestamp_ms",
    "generate_checksummed_id",
    "generate_object_id",
    "generate_stable_content_id",
    "generate_stable_uuid5",
    "generate_ulid",
    "generate_uuid7",
    "verify_checksummed_id",
]

from coola.identifier.checksummed import generate_checksummed_id, verify_checksummed_id
from coola.identifier.content import generate_stable_content_id, generate_stable_uuid5
from coola.identifier.objectid import (
    extract_object_id_timestamp_s,
    generate_object_id,
)
from coola.identifier.snowflake import (
    SnowflakeIdGenerator,
    extract_snowflake_timestamp_ms,
)
from coola.identifier.ulid import extract_ulid_timestamp_ms, generate_ulid
from coola.identifier.uuid7 import extract_uuid7_timestamp_ms, generate_uuid7
