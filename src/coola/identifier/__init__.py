r"""Provide identifiers for nested data.

Every generator returns a ``str`` identifier except
``generate_snowflake_id``/``SnowflakeIdGenerator.generate``, which
return a plain ``int`` (a Snowflake ID is defined as a 64-bit integer,
e.g. for use as a database ``BIGINT`` primary key).
"""

from __future__ import annotations

__all__ = [
    "ObjectIdGenerator",
    "SnowflakeIdGenerator",
    "decode_obfuscated_id",
    "extract_object_id_timestamp",
    "extract_snowflake_timestamp_ms",
    "extract_ulid_timestamp_ms",
    "extract_uuid7_timestamp_ms",
    "generate_checksummed_id",
    "generate_nano_id",
    "generate_obfuscated_id",
    "generate_object_id",
    "generate_prefixed_id",
    "generate_snowflake_id",
    "generate_stable_content_id",
    "generate_stable_uuid5",
    "generate_ulid",
    "generate_uuid4",
    "generate_uuid7",
    "verify_checksummed_id",
]

from coola.identifier.checksummed import generate_checksummed_id, verify_checksummed_id
from coola.identifier.content import generate_stable_content_id
from coola.identifier.nanoid import generate_nano_id
from coola.identifier.obfuscated import decode_obfuscated_id, generate_obfuscated_id
from coola.identifier.objectid import (
    ObjectIdGenerator,
    extract_object_id_timestamp,
    generate_object_id,
)
from coola.identifier.prefixed import generate_prefixed_id
from coola.identifier.snowflake import (
    SnowflakeIdGenerator,
    extract_snowflake_timestamp_ms,
    generate_snowflake_id,
)
from coola.identifier.ulid import extract_ulid_timestamp_ms, generate_ulid
from coola.identifier.uuid4 import generate_uuid4
from coola.identifier.uuid5 import generate_stable_uuid5
from coola.identifier.uuid7 import extract_uuid7_timestamp_ms, generate_uuid7
