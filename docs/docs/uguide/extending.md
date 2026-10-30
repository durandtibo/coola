# How to Add Support for a New Type

:book: This tutorial walks through teaching `coola` about one custom class by registering an
**equality tester**, a **summarizer**, and a **hasher**. Each one is registered in the default
registry of its package, so the top-level functions (`objects_are_equal`, `summarize`,
`hash_object`) pick it up automatically.

**Prerequisites:** You'll need to know a bit of Python.

## The custom class

```pycon
>>> class Point:
...     def __init__(self, x: float, y: float) -> None:
...         self.x = x
...         self.y = y

```

## 1. Equality

Subclass `BaseEqualityTester`, implement `objects_are_equal`, then register it. A tester never
recurses: it returns a verdict (`True`, or `ctx.fail(reason)` which returns `False`), or the
`(actual, expected, path part)` pairs that must all be equal. The engine checks the identity and the
type, applies the tolerance, detects the cycles and locates the difference.

```pycon
>>> from coola.equality import objects_are_equal
>>> from coola.equality import AttributeName
>>> from coola.equality.tester import BaseEqualityTester, register_equality_testers
>>> class PointEqualityTester(BaseEqualityTester[Point]):
...     def objects_are_equal(self, actual: Point, expected: Point, ctx):
...         return (
...             (actual.x, expected.x, AttributeName("x")),
...             (actual.y, expected.y, AttributeName("y")),
...         )
>>> register_equality_testers({Point: PointEqualityTester()})
>>> objects_are_equal(Point(1, 2), Point(1, 2))
True
>>> objects_are_equal(Point(1, 2), Point(1, 3))
False

```

## 2. Summary

Subclass `BaseSummarizer` and describe one type: return a `str` for a leaf, or a `Node` (a header and
the children) for a container. The summarizer does not recurse: the engine summarizes the children
with the summarizer of their own type.

```pycon
>>> from coola.summary import BaseSummarizer, SummaryConfig, register_summarizers, summarize
>>> class PointSummarizer(BaseSummarizer[Point]):
...     def summarize(self, data: Point, config: SummaryConfig, *, compact: bool) -> str:
...         return f"Point(x={data.x}, y={data.y})"
>>> register_summarizers({Point: PointSummarizer()})
>>> print(summarize(Point(1, 2)))
Point(x=1, y=2)

```

## 3. Hashing

Subclass `BaseHasher`. A hasher only describes the content of a type: `encode` returns a `Node`
(a tag and the children to hash) or, for a leaf, `tagged(tag, payload)`. The engine hashes the
children with the hasher of their own type, so the hasher never receives the registry or the
options of `hash_object`. A hasher must be consistent with equality: objects that compare equal
must produce the same hash.

```pycon
>>> from coola.hashing import BaseHasher, Node, hash_object, register_hashers
>>> class PointHasher(BaseHasher[Point]):
...     def encode(self, data: Point) -> Node:
...         return Node("Point", [data.x, data.y])
>>> register_hashers({Point: PointHasher()})
>>> hash_object(Point(1, 2)) == hash_object(Point(1, 2))
True

```

## Notes

- Registries resolve by type using the MRO, so a tester registered for a base class also handles
  its subclasses unless a more specific one is registered.
- Registering an already-registered type raises an error unless `exist_ok=True`.
- Custom types nested in lists or dicts work out of the box once registered.

## Registering types from a third-party package

A package can add support for its own types without users calling
`register_*` by hand. Expose a function that takes the registry and
registers types on it, then declare it as an entry point in one of these
groups:

| Group                          | Registry               |
| ------------------------------ | ---------------------- |
| `coola.equality.testers`       | equality testers       |
| `coola.summary.summarizers`    | summarizers            |
| `coola.hashing.hashers`        | hashers                |
| `coola.recursive.transformers` | recursive transformers |
| `coola.jsonable.transformers`  | JSON transformers      |
| `coola.iterator.child_finders` | Iterator child finders |
| `coola.random.managers`        | random managers        |

```toml
# pyproject.toml of the plugin package
[project.entry-points."coola.summary.summarizers"]
mypkg = "mypkg.coola_plugin:register"
```

```python
# mypkg/coola_plugin.py
def register(registry):
    registry.register_many({MyType: MySummarizer()})
```

Plugins run once, when the default registry is first built, after the
built-in entries. A plugin that raises is skipped with a `RuntimeWarning`, and the failure is logged
with its traceback. Set the `COOLA_STRICT_PLUGINS=1` environment variable (e.g. in CI)
to raise the failures instead, so a broken plugin is not missed: every plugin is still attempted,
and all the failures are raised together as an `ExceptionGroup`, each annotated with the name and
the group of the plugin.

Set `COOLA_DISABLE_PLUGINS=1` to load no plugin at all (a reproducible run, or to check whether a
plugin causes a conflict). To find out which plugins were loaded in the process, for example when
`objects_are_equal` or a hash is not what you expect, call `get_loaded_plugins()`:

```pycon
>>> from coola.equality.tester import get_default_registry
>>> from coola.registry.plugins import get_loaded_plugins
>>> get_default_registry()  # doctest: +SKIP
>>> [(plugin.group, plugin.name, plugin.value) for plugin in get_loaded_plugins()]  # doctest: +SKIP
[('coola.equality.testers', 'mypkg', 'mypkg.coola_plugin:register')]

```

A plugin is listed after it was applied to a default registry, which loads its plugins the first
time it is built.
