# How to Add Support for a New Type

:book: This tutorial walks through teaching `coola` about one custom class by registering an
**equality tester**, a **summarizer**, and a **hasher**. Each one is registered in the default
registry of its package, so the top-level functions (`objects_are_equal`, `summary`,
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

Subclass `BaseEqualityTester`, implement `equal` (used to compare testers) and
`objects_are_equal`, then register it.

```pycon
>>> from coola.equality import objects_are_equal
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import BaseEqualityTester, register_equality_testers
>>> class PointEqualityTester(BaseEqualityTester[Point]):
...     def equal(self, other: object) -> bool:
...         return type(other) is type(self)
...
...     def objects_are_equal(self, actual: Point, expected: Point, config: EqualityConfig) -> bool:
...         return config.registry.objects_are_equal(
...             (actual.x, actual.y), (expected.x, expected.y), config
...         )
>>> register_equality_testers({Point: PointEqualityTester()})
>>> objects_are_equal(Point(1, 2), Point(1, 2))
True
>>> objects_are_equal(Point(1, 2), Point(1, 3))
False

```

## 2. Summary

Subclass `BaseSummarizer`; the `registry` argument lets you summarize nested values.

```pycon
>>> from coola.summary import BaseSummarizer, SummarizerRegistry, register_summarizers, summarize
>>> class PointSummarizer(BaseSummarizer[Point]):
...     def equal(self, other: object) -> bool:
...         return type(other) is type(self)
...
...     def summarize(
...         self, data: Point, registry: SummarizerRegistry, depth: int = 0, max_depth: int = 1
...     ) -> str:
...         return f"Point(x={data.x}, y={data.y})"
>>> register_summarizers({Point: PointSummarizer()})
>>> print(summarize(Point(1, 2)))
Point(x=1, y=2)

```

## 3. Hashing

Subclass `BaseHasher`. A hasher must be consistent with equality: objects that compare equal
must produce the same hash.

```pycon
>>> from coola.hashing import BaseHasher, HasherRegistry, hash_object, register_hashers
>>> class PointHasher(BaseHasher[Point]):
...     def equal(self, other: object) -> bool:
...         return type(other) is type(self)
...
...     def hash(
...         self,
...         data: Point,
...         registry: HasherRegistry,
...         length: int = 64,
...         ignore_unhashable: bool = False,
...     ) -> str:
...         return registry.hash((data.x, data.y), length=length)
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
| `coola.iterator.dfs`           | DFS iterators          |
| `coola.iterator.bfs`           | BFS child finders      |
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
to re-raise the exception instead, so a broken plugin is not missed.
