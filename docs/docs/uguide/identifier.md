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
| `generate_stable_uuid`          | :white_check_mark:    | UUID string           | Reproducible ID, needs a valid UUID (e.g. UUID DB column)      |
| `generate_stable_content_id`    | :white_check_mark:    | hex string            | Reproducible ID, full hash strength (dedup, caching)           |
| `generate_ulid`                 | :x:                   | 26-char string         | Unique, sortable-by-creation-time ID                           |
| `generate_uuid7`                | :x:                   | UUID string            | Unique, sortable-by-creation-time ID, needs a valid UUID       |
| `generate_snowflake_id`         | :x:                   | 64-bit integer        | Unique, sortable-by-creation-time ID as a single integer        |
| `generate_prefixed_id`          | depends on generator  | `"{prefix}_{id}"`     | Wraps any of the above so the ID's type is recognizable at a glance |

`generate_stable_uuid` and `generate_stable_content_id` are content-addressed: they are built on
top of [`coola.hashing`](../refs/hashing.md)'s `hash_object`, so calling them twice with equal data
(regardless of e.g. mapping insertion order) always returns the same identifier. `generate_ulid`,
`generate_uuid7`, and `generate_snowflake_id` are not derived from data at all: they mint a fresh,
unique value every call, ordered by creation time instead. `generate_prefixed_id` is not a new
algorithm; it wraps any of the others (or a custom callable) to tag the identifier's type.

## Stable, content-derived identifiers

### `generate_stable_uuid`

`generate_stable_uuid` computes a stable, reproducible UUID for a nested data structure. It hashes
`data` with `hash_object` (so mapping key order, for example, does not affect the result), then
derives a deterministic `uuid.uuid5` from that digest under a fixed namespace:

```pycon
>>> from coola.identifier import generate_stable_uuid
>>> generate_stable_uuid({"source": "cats.txt", "page": 1})  # doctest: +ELLIPSIS
'...'
>>> generate_stable_uuid({"page": 1, "source": "cats.txt"}) == generate_stable_uuid(
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
>>> from coola.identifier import generate_stable_uuid
>>> namespace = uuid.uuid4()
>>> generate_stable_uuid({"a": 1}, namespace=namespace)  # doctest: +ELLIPSIS
'...'

```

!!! warning

    The UUID `generate_stable_uuid` returns for a given input is stable only as long as
    `hash_object` (and the hashers resolved for the types in the data) keep producing the same
    digest. Do not rely on cross-version stability for UUIDs persisted long-term unless you pin
    `coola` and pass an explicit, version-controlled hasher registry.

### `generate_stable_content_id`

`generate_stable_content_id` is an alternative to `generate_stable_uuid` that skips the UUID
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

Prefer `generate_stable_content_id` over `generate_stable_uuid` when you don't need UUID format
compliance (e.g. internal cache keys, deduplication) and want the strongest possible collision
resistance.

### Custom hasher registry and unhashable data

Both `generate_stable_uuid` and `generate_stable_content_id` forward `registry` and
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

- Need the same identifier every time for the same data? Use `generate_stable_uuid` (valid UUID
  format) or `generate_stable_content_id` (raw hash, stronger collision resistance, configurable
  length).
- Need a unique identifier per call, sortable by creation time? Use `generate_ulid` (string,
  more randomness), `generate_uuid7` (string, valid UUID format), or `generate_snowflake_id`
  (64-bit integer).
- Need the identifier's type recognizable at a glance (e.g. in logs or URLs)? Wrap any of the
  above with `generate_prefixed_id`.
