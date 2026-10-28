# Identifiers

:book: This page describes the `coola.identifier` package, which provides functions to generate
identifiers for nested data structures and for time-ordered records.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.identifier` package provides several identifier generators, covering two different
problems:

| Function                     |  Same data → same ID?  | Format              | Use case                                                                         |
| ---------------------------- | :--------------------: | ------------------- | -------------------------------------------------------------------------------- |
| `generate_stable_uuid5`      |   :white_check_mark:   | UUID string         | Reproducible ID, needs a valid UUID (e.g. UUID DB column)                        |
| `generate_stable_content_id` |   :white_check_mark:   | hex string          | Reproducible ID, full hash strength (dedup, caching)                             |
| `generate_ulid`              |          :x:           | 26-char string      | Unique, sortable-by-creation-time ID                                             |
| `generate_uuid7`             |          :x:           | UUID string         | Unique, sortable-by-creation-time ID, needs a valid UUID                         |
| `generate_snowflake_id`      |          :x:           | 64-bit integer      | Unique, sortable-by-creation-time ID as a single integer                         |
| `generate_object_id`         |          :x:           | 24-char hex string  | Unique, sortable-by-creation-time ID, never blocks/raises                        |
| `generate_checksummed_id`    |          :x:           | grouped string      | Random ID with a check symbol, for identifiers humans retype                     |

`generate_stable_uuid5` and `generate_stable_content_id` are content-addressed: they are built on
top of [`coola.hashing`](../refs/hashing.md)'s `hash_object`, so calling them twice with equal data
(regardless of e.g. mapping insertion order) always returns the same identifier. `generate_ulid`,
`generate_uuid7`, `generate_snowflake_id` and `generate_object_id` are not derived from data at all:
they mint a fresh, unique value every call, ordered by creation time instead.

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
column. A UUID has 122 random bits, so its collision resistance is bounded by its width (birthday
bound 2^61), regardless of how strong the underlying hash used by `hash_object` is.

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

    The hash format changed in `coola` 2.0 (see the [migration guide](../migration/v1.1_to_v2.0.md)):
    every identifier changes, and an identifier stored with an older version must be regenerated.

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
>>> generate_stable_content_id({"page": 1, "source": "cats.txt"}) == generate_stable_content_id(
...     {"source": "cats.txt", "page": 1}
... )
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
>>> import warnings
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")  # a warning says that the type has no hasher
...     generate_stable_content_id(object(), ignore_unhashable=True)  # doctest: +ELLIPSIS
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

ULIDs generated with the current time are **monotonic within a process**, as recommended by the ULID
specification: ULIDs generated in the same millisecond (or after the clock moved backward) keep the
timestamp and increment the random part by one, so they sort in the order of generation. Across
processes, ULIDs generated in the same millisecond are not ordered. A ULID generated with an explicit
`timestamp_ms` always has a random part, so it is not ordered relative to the others.

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

Like ULIDs, UUIDv7 values sort (as plain strings) in creation-time order, and they are monotonic
within a process when the current time is used (the 74 random bits are incremented within the same
millisecond), which keeps the inserts in a database index append-only. Prefer `generate_uuid7`
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
avoid collisions between them. This includes the children of a fork (`os.fork`, or `multiprocessing`
with the `fork` start method): they inherit the state of the generator of their parent, so they must
not keep the default `worker_id=0`.

If the system clock moves backward (typically a step adjustment by NTP, or a virtual machine that
resumes from a pause), the generator raises a `RuntimeError` instead of minting an ID that could be a
duplicate or smaller than a previous one. This is a deliberate choice: waiting for the clock to catch
up could block for an unbounded time. The error is raised before the state of the generator is
modified, so the next call works once the clock is back to the last timestamp. If your system can
see steps of the clock, retry the call after a short delay, or run the time synchronization in
slew mode.

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

### `generate_object_id`

`generate_object_id` generates a MongoDB `ObjectId` style 24-character hex string: a 4-byte
timestamp, a 5-byte per-process value, and a 3-byte counter. Like `SnowflakeIdGenerator`, later IDs
sort after earlier ones, but the counter silently wraps instead of raising or blocking when
exhausted within a second, and no `worker_id` needs to be configured. After a fork, the child
process gets its own random per-process value and counter, so the parent and its children do not
produce the same identifiers:

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

## Properties

| Property                       | Statement                                                                                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Declared shape                 | Every generator returns a `str`, except `SnowflakeIdGenerator.generate` and `generate_snowflake_id`, which return an `int` of at most 63 bits.                                      |
| Random bits are CSPRNG         | The random bits of `generate_ulid`, `generate_uuid7`, `generate_object_id` and `generate_checksummed_id` come from `secrets` or `os.urandom`, never `random`.   |
| Identifier, not a secret       | An identifier is not an access token. The time-ordered ones are guessable (the next value is the previous one plus one) and the content-derived ones are reproducible. Authorize every access explicitly. |
| Stable IDs are content-derived | `generate_stable_uuid5` and `generate_stable_content_id` give the same ID for equal data, regardless of the mapping insertion order.                                                |
| Collision resistance           | `generate_stable_uuid5` is bounded by the 122 random bits of a UUID (birthday bound 2^61), whatever the hash. `generate_stable_content_id` keeps the strength of the underlying hash. |
| Time-ordered IDs sort          | ULID, UUIDv7 and Snowflake IDs sort in creation order. They are monotonic within a process when the current time is used. `generate_object_id` sorts by second only.               |
| Not ordered across processes   | Two processes that mint an ID in the same tick (a millisecond, a second for ObjectId) do not give ordered IDs.                                                                      |
| Snowflake uniqueness           | It holds for the calls that share a generator. Give each concurrent generator, including each fork, its own `worker_id`.                                                            |
| Thread-safe                    | Every generator can be called from several threads without a lock of the caller.                                                                                                    |
| Fork-safe                      | After a fork, `generate_ulid`, `generate_uuid7` and `generate_object_id` do not repeat the identifiers of the parent. A Snowflake generator does, unless the child uses another `worker_id`. |
| Recoverable timestamp          | The four time-ordered generators have a matching `extract_*` function.                                                                                                              |
| Check symbol                   | `verify_checksummed_id` is `True` for every identifier of `generate_checksummed_id` and detects any single wrong character and any swap of two adjacent characters.                  |

The stable IDs inherit the [hashing](hashing.md) properties, including the fact that they can change between versions of `coola`.

Not guaranteed:

- **Order across processes.** Identifiers minted in the same tick by two processes are ordered by
  their random part, not by the time of the call.
- **Uniqueness of Snowflake IDs across processes** that use the same `worker_id`: assigning worker
  numbers is the job of the caller.
- **Stability of the stable IDs across major versions of `coola`**, and across registries other than
  the stable one.
- **Unguessability** of any identifier.

## Which one should I use?

- Need the same identifier every time for the same data? Use `generate_stable_uuid5` (valid UUID
  format) or `generate_stable_content_id` (raw hash, stronger collision resistance, configurable
  length).
- Need a unique identifier per call, sortable by creation time? Use `generate_ulid` (string,
  more randomness), `generate_uuid7` (string, valid UUID format), `generate_snowflake_id`
  (64-bit integer) or `generate_object_id` (24-character hex string, sorts by second).
- Need a code that a human retypes (support code, license key)? Use `generate_checksummed_id`
  and `verify_checksummed_id`.
- Need a plain random UUID, a short random string, a prefixed identifier or an opaque public
  identifier? Use the standard library or a dedicated library: `str(uuid.uuid4())`,
  `secrets.token_urlsafe(16)`, `f"cus_{generate_ulid()}"`, or a library such as `sqids`.
- Need an unpredictable token (session, API key, password reset link)? None of these: use
  `secrets.token_urlsafe`.
