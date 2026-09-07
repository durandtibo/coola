# Identifiers

:book: This page describes the `coola.identifier` package, which provides functions to generate
identifiers for nested data structures and for time-ordered records.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.identifier` package provides several identifier generators, covering two different
problems:

| Function                       | Same data → same ID? | Format               | Use case                                                    |
|---------------------------------|:---------------------:|-----------------------|---------------------------------------------------------------|
| `generate_stable_uuid5`          | :white_check_mark:    | UUID string           | Reproducible ID, needs a valid UUID (e.g. UUID DB column)      |
| `generate_stable_content_id`    | :white_check_mark:    | hex string            | Reproducible ID, full hash strength (dedup, caching)           |
| `generate_ulid`                 | :x:                   | 26-char string         | Unique, sortable-by-creation-time ID                           |
| `generate_uuid7`                | :x:                   | UUID string            | Unique, sortable-by-creation-time ID, needs a valid UUID       |
| `generate_snowflake_id`         | :x:                   | 64-bit integer        | Unique, sortable-by-creation-time ID as a single integer        |
| `generate_object_id`            | :x:                   | 24-char hex string    | Unique, sortable-by-creation-time ID, never blocks/raises        |
| `generate_uuid4`                | :x:                   | UUID string            | Plain random ID, needs a valid UUID, no ordering                |
| `generate_nano_id`              | :x:                   | configurable string   | Short, URL-safe random ID with custom alphabet/length            |
| `generate_checksummed_id`       | :x:                   | grouped string         | Random ID with a check symbol, for identifiers humans retype     |
| `generate_obfuscated_id`        | reversible, not random| base62 string          | Obfuscates an existing sequential integer, decodable with `decode_obfuscated_id` |
| `generate_prefixed_id`          | depends on generator  | `"{prefix}_{id}"`     | Wraps any of the above so the ID's type is recognizable at a glance |

`generate_stable_uuid5` and `generate_stable_content_id` are content-addressed: they are built on
top of [`coola.hashing`](../refs/hashing.md)'s `hash_object`, so calling them twice with equal data
(regardless of e.g. mapping insertion order) always returns the same identifier. `generate_ulid`,
`generate_uuid7`, and `generate_snowflake_id` are not derived from data at all: they mint a fresh,
unique value every call, ordered by creation time instead. `generate_prefixed_id` is not a new
algorithm; it wraps any of the others (or a custom callable) to tag the identifier's type.

The four time-ordered generators (`generate_ulid`, `generate_uuid7`, `generate_snowflake_id`, and
`generate_object_id`) each have a matching `extract_*_timestamp*` function
(`extract_ulid_timestamp_ms`, `extract_uuid7_timestamp_ms`, `extract_snowflake_timestamp_ms`, and
`extract_object_id_timestamp`) that recovers the timestamp packed into a previously generated
identifier.

## Stable, content-derived identifiers

### `generate_stable_uuid5`

`generate_stable_uuid5` computes a stable, reproducible UUID for a nested data structure. It hashes
`data` with `hash_object` (so mapping key order, for example, does not affect the result), then
derives a deterministic `uuid.uuid5` from that digest under a fixed namespace:

```pycon
>>> from coola.identifier import generate_stable_uuid5
>>> generate_stable_uuid5({"source": "cats.txt", "page": 1})  # doctest: +ELLIPSIS
'...'
>>> generate_stable_uuid5({"page": 1, "source": "cats.txt"}) == generate_stable_uuid5(
...     {"source": "cats.txt", "page": 1}
... )
True

```

Use it when the identifier must be a valid UUID string, e.g. to populate a UUID-typed database
column. Because `uuid.uuid5` hashes its input with SHA-1 internally, the resulting UUID's
collision resistance is bounded by SHA-1, regardless of how strong the underlying hash used by
`hash_object` is.

Pass a custom `namespace` to derive UUIDs in a separate identifier space, e.g. to avoid collisions
with UUIDs minted by another system for unrelated data:

```pycon
>>> import uuid
>>> from coola.identifier import generate_stable_uuid5
>>> namespace = uuid.uuid4()
>>> generate_stable_uuid5({"a": 1}, namespace=namespace)  # doctest: +ELLIPSIS
'...'

```

!!! warning

    The UUID `generate_stable_uuid5` returns for a given input is stable only as long as
    `hash_object` (and the hashers resolved for the types in the data) keep producing the same
    digest. Do not rely on cross-version stability for UUIDs persisted long-term unless you pin
    `coola` and pass an explicit, version-controlled hasher registry.

### `generate_stable_content_id`

`generate_stable_content_id` is an alternative to `generate_stable_uuid5` that skips the UUID
reshaping step: it returns the `hash_object` digest directly.

```pycon
>>> from coola.identifier import generate_stable_content_id
>>> generate_stable_content_id({"source": "cats.txt", "page": 1})  # doctest: +ELLIPSIS
'...'
>>> generate_stable_content_id(
...     {"page": 1, "source": "cats.txt"}
... ) == generate_stable_content_id({"source": "cats.txt", "page": 1})
True

```

This keeps the full collision resistance and configurable `length` of the underlying hash, at the
cost of not being a valid UUID string:

```pycon
>>> from coola.identifier import generate_stable_content_id
>>> generate_stable_content_id({"a": 1}, length=16)  # doctest: +ELLIPSIS
'...'

```

Prefer `generate_stable_content_id` over `generate_stable_uuid5` when you don't need UUID format
compliance (e.g. internal cache keys, deduplication) and want the strongest possible collision
resistance.

### Custom hasher registry and unhashable data

Both `generate_stable_uuid5` and `generate_stable_content_id` forward `registry` and
`ignore_unhashable` to `hash_object`:

```pycon
>>> from coola.hashing import HasherRegistry, StringHasher
>>> from coola.identifier import generate_stable_content_id
>>> registry = HasherRegistry({object: StringHasher()})
>>> generate_stable_content_id("meow", registry=registry)  # doctest: +ELLIPSIS
'...'
>>> generate_stable_content_id(object(), ignore_unhashable=True)  # doctest: +ELLIPSIS
'...'

```

By default, a `KeyError` is raised if `data` contains a type for which no hasher is registered.
See the [hashing reference](../refs/hashing.md) for how to register hashers for custom types.

## Non-deterministic, time-ordered identifiers

### `generate_ulid`

`generate_ulid` generates a
[ULID](https://github.com/ulid/spec) (Universally Unique Lexicographically Sortable Identifier):
a 26-character Crockford Base32 string packing a 48-bit millisecond timestamp followed by 80 bits
of randomness. Two calls with the same input never return the same value:

```pycon
>>> from coola.identifier import generate_ulid
>>> generate_ulid()  # doctest: +ELLIPSIS
'...'
>>> generate_ulid() == generate_ulid()
False

```

Because the timestamp is the most significant part, ULIDs generated later sort (as plain strings)
after ULIDs generated earlier, unlike `uuid.uuid4`, which sorts randomly. Use it for a unique
record ID that should also sort roughly by insertion order.

Use `extract_ulid_timestamp_ms` to recover the timestamp encoded in a ULID, e.g. to check how old a
record is without a separate stored timestamp column:

```pycon
>>> from coola.identifier import extract_ulid_timestamp_ms, generate_ulid
>>> ulid = generate_ulid(timestamp_ms=1704067200000)
>>> extract_ulid_timestamp_ms(ulid)
1704067200000

```

### `generate_uuid7`

`generate_uuid7` generates a [UUIDv7](https://www.rfc-editor.org/rfc/rfc9562) (RFC 9562): like
`generate_ulid`, it packs a 48-bit millisecond timestamp followed by randomness, but into the
standard 128-bit UUID layout (version and variant bits included) instead of a Base32 string:

```pycon
>>> from coola.identifier import generate_uuid7
>>> generate_uuid7()  # doctest: +ELLIPSIS
'...'
>>> generate_uuid7() == generate_uuid7()
False

```

Like ULIDs, UUIDv7 values sort (as plain strings) in creation-time order. Prefer `generate_uuid7`
over `generate_ulid` when the identifier must be a valid UUID string (e.g. a UUID-typed database
column, or an API expecting `uuid.UUID` formatting); prefer `generate_ulid` otherwise, since it
packs more randomness (80 bits) than UUIDv7 leaves available (74 bits, once the version and
variant bits are subtracted).

`extract_uuid7_timestamp_ms` recovers the timestamp, the same way `extract_ulid_timestamp_ms` does
for a ULID:

```pycon
>>> from coola.identifier import extract_uuid7_timestamp_ms, generate_uuid7
>>> uuid7 = generate_uuid7(timestamp_ms=1704067200000)
>>> extract_uuid7_timestamp_ms(uuid7)
1704067200000

```

### `generate_snowflake_id`

`generate_snowflake_id` generates a Snowflake-style 64-bit integer identifier, in the spirit of
Twitter's original Snowflake service: a 41-bit millisecond timestamp, a 10-bit `worker_id`, and a
12-bit per-millisecond sequence number, packed into a single integer:

```pycon
>>> from coola.identifier import generate_snowflake_id
>>> id1 = generate_snowflake_id()
>>> id2 = generate_snowflake_id(worker_id=3)

```

Like `generate_ulid`, successive IDs are monotonically increasing, but the result is a plain
64-bit integer rather than a string, useful when the identifier must fit a `BIGINT`-style column,
or when IDs need to be attributable to the worker/shard that minted them via `worker_id`.

`generate_snowflake_id` is a thread-safe convenience wrapper around a shared, process-wide
`SnowflakeIdGenerator` instance: its sequence counter is local to that instance, so it guarantees
uniqueness across calls sharing it, not across other instances or processes. Assign each
concurrently running generator (typically one per process or shard) a distinct `worker_id` to
avoid collisions between them.

Use `SnowflakeIdGenerator` directly instead of the module-level function when you need several
independent generators in the same process, e.g. one per worker thread, or an isolated instance
in a test, without them sharing state through a global singleton:

```pycon
>>> from coola.identifier import SnowflakeIdGenerator
>>> generator = SnowflakeIdGenerator()
>>> id1 = generator.generate()
>>> id2 = generator.generate(worker_id=3)

```

`extract_snowflake_timestamp_ms` recovers the timestamp encoded in a Snowflake-style ID:

```pycon
>>> from coola.identifier import extract_snowflake_timestamp_ms, generate_snowflake_id
>>> snowflake_id = generate_snowflake_id()
>>> isinstance(extract_snowflake_timestamp_ms(snowflake_id), int)
True

```

### `generate_uuid4`

`generate_uuid4` is a thin wrapper around `uuid.uuid4()`, provided for API symmetry with
`generate_uuid7` and `generate_stable_uuid5`. It carries no timestamp and does not sort by
creation time:

```pycon
>>> from coola.identifier import generate_uuid4
>>> generate_uuid4()  # doctest: +ELLIPSIS
'...'

```

### `generate_nano_id`

`generate_nano_id` generates a [Nano ID](https://github.com/ai/nanoid) style random string: unlike
`generate_ulid` and `generate_uuid7`, both its alphabet and its length are configurable, which
makes it a better fit for short, URL-safe identifiers (e.g. slugs):

```pycon
>>> from coola.identifier import generate_nano_id
>>> generate_nano_id()  # doctest: +ELLIPSIS
'...'
>>> generate_nano_id(size=8, alphabet="0123456789abcdef")  # doctest: +ELLIPSIS
'...'

```

### `generate_object_id`

`generate_object_id` generates a MongoDB `ObjectId` style 24-character hex string: a 4-byte
timestamp, a 5-byte per-process value, and a 3-byte counter. Like `SnowflakeIdGenerator`, later IDs
sort after earlier ones, but the counter silently wraps instead of raising or blocking when
exhausted within a second, and no `worker_id` needs to be configured:

```pycon
>>> from coola.identifier import generate_object_id
>>> generate_object_id()  # doctest: +ELLIPSIS
'...'

```

`extract_object_id_timestamp` recovers the (second-resolution) Unix timestamp encoded in an
identifier previously returned by `generate_object_id` or `ObjectIdGenerator.generate`:

```pycon
>>> from coola.identifier import extract_object_id_timestamp, generate_object_id
>>> object_id = generate_object_id()
>>> isinstance(extract_object_id_timestamp(object_id), int)
True

```

### `generate_checksummed_id`

`generate_checksummed_id` adds a Crockford Base32 check symbol to a random identifier, so that a
single mistyped or transposed character is caught locally, without needing a lookup. Use it for
identifiers a human is expected to read back or retype, e.g. a support code or license key:

```pycon
>>> from coola.identifier import generate_checksummed_id, verify_checksummed_id
>>> checksummed_id = generate_checksummed_id()
>>> verify_checksummed_id(checksummed_id)
True

```

`verify_checksummed_id` follows the Crockford Base32 spec's own transcription rules: it is
case-insensitive, and it normalizes `'O'` to `'0'` and `'I'`/`'L'` to `'1'` before checking the
check symbol, so a human who reads back `'O'` for `'0'` (or types in lowercase) still verifies
correctly:

```pycon
>>> verify_checksummed_id(checksummed_id.lower())
True

```

### `generate_obfuscated_id` and `decode_obfuscated_id`

Unlike every other generator above, `generate_obfuscated_id` does not mint a new value: it takes
an existing non-negative integer (e.g. a database autoincrement ID) and reshapes it into a short
opaque string that hides its magnitude and ordering, while remaining exactly reversible via
`decode_obfuscated_id` given the same `salt`:

```pycon
>>> from coola.identifier import generate_obfuscated_id, decode_obfuscated_id
>>> encoded = generate_obfuscated_id(42, salt="orders")
>>> decode_obfuscated_id(encoded, salt="orders")
42

```

This is obfuscation, not encryption: do not rely on it to hide data from anyone who can observe
many `(plaintext, obfuscated)` pairs for a known `salt`.

## Prefixed identifiers

### `generate_prefixed_id`

`generate_prefixed_id` is not a new identifier-generation algorithm: it wraps any of the
generators above (via its `generator` argument, defaulting to `generate_ulid`) to produce
Stripe-style prefixed identifiers, which make an identifier's type recognizable at a glance (e.g.
in logs, URLs, or support tickets) without a lookup:

```pycon
>>> from coola.identifier import generate_prefixed_id
>>> generate_prefixed_id("cus")  # doctest: +ELLIPSIS
'cus_...'

```

Pass a different `generator` to prefix a different kind of identifier, e.g. a `SnowflakeIdGenerator`
instance's output turned into a string:

```pycon
>>> from coola.identifier import SnowflakeIdGenerator, generate_prefixed_id
>>> generator = SnowflakeIdGenerator()
>>> generate_prefixed_id("evt", generator=lambda: str(generator.generate()))
... # doctest: +ELLIPSIS
'evt_...'

```

`prefix` must be non-empty and must not contain `"_"` (the separator between the prefix and the
generated identifier):

```pycon
>>> from coola.identifier import generate_prefixed_id
>>> generate_prefixed_id("")
Traceback (most recent call last):
    ...
ValueError: prefix must not be empty
>>> generate_prefixed_id("cus_tom")
Traceback (most recent call last):
    ...
ValueError: prefix must not contain '_', got 'cus_tom'

```

## Which one should I use?

- Need the same identifier every time for the same data? Use `generate_stable_uuid5` (valid UUID
  format) or `generate_stable_content_id` (raw hash, stronger collision resistance, configurable
  length).
- Need a unique identifier per call, sortable by creation time? Use `generate_ulid` (string,
  more randomness), `generate_uuid7` (string, valid UUID format), or `generate_snowflake_id`
  (64-bit integer).
- Need the identifier's type recognizable at a glance (e.g. in logs or URLs)? Wrap any of the
  above with `generate_prefixed_id`.
