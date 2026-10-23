# Factory Utilities

:book: This page describes the `coola.factory` package, which provides utilities to
dynamically import and instantiate Python objects from configuration dictionaries.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.factory` package makes it possible to build objects (classes or functions) from plain
data, such as a dictionary loaded from JSON or YAML. This is a common pattern for configuration-
driven code: instead of hard-coding which class to instantiate, you describe it with a
fully-qualified name and a set of keyword arguments.

The package provides:

- **`import_object`**: import an object (class, function, ...) from its fully qualified name
- **`instantiate_object`**: instantiate an object given a target and its arguments
- **`factory`**: a universal factory function that imports and instantiates an object from a
  fully qualified name, without requiring prior registration
- **`resolve_object`**: return an object as-is, or build it from a factory configuration
- **`is_object_config`**: check if a configuration dictionary targets a given class
- **`OBJECT_TARGET`** / **`OBJECT_INIT`**: the configuration keys used by the functions above

Unlike [`coola.registry`](registry.md), which requires classes to be registered under a key
before they can be looked up, `coola.factory` resolves objects directly from their fully
qualified Python path, so any importable object can be used without registration.

## Importing an object by name

`import_object` dynamically imports a class, function, or other object using its fully
qualified name (`module_path.object_name`):

```pycon

>>> from coola.factory import import_object
>>> cls = import_object("collections.Counter")
>>> cls()
Counter()
>>> fn = import_object("math.isclose")
>>> fn(1.0, 1.0)
True

```

It raises a `TypeError` if `object_path` is not a string, and an `ImportError` if the path
cannot be resolved.

## Instantiating an object from a configuration

`factory` combines `import_object` with instantiation. It takes a `_target_` key with the
fully qualified name of a class or function, and forwards the remaining arguments to it:

```pycon

>>> from coola.factory import factory
>>> factory("collections.Counter", [1, 2, 1, 3])
Counter({1: 2, 2: 1, 3: 1})

```

Because configurations are often expressed as dictionaries (e.g. parsed from JSON or YAML), it
is common to unpack the configuration into `factory`. The `OBJECT_TARGET` constant
(`"_target_"`) is provided so that code does not need to hard-code the literal key name:

```pycon

>>> from coola.factory import factory, OBJECT_TARGET
>>> config = {OBJECT_TARGET: "collections.Counter", "a": 4, "b": 2}
>>> factory(**config)
Counter({'a': 4, 'b': 2})

```

### Controlling how the object is created

By default, `factory` calls the target's `__init__`. The special `OBJECT_INIT` key
(`"_init_"`) can be set to `"__new__"`, or to the name of a class method, to control how the
object is created instead:

```pycon

>>> from coola.factory import factory, OBJECT_INIT
>>> factory("collections.OrderedDict", _init_="__init__", a=1)
OrderedDict({'a': 1})

```

## Resolving an object or a configuration

`resolve_object` is convenient in code that accepts either an already-built instance or a
configuration dictionary. If the input is already an instance of the expected type, it is
returned unchanged; if it is a `dict`, it is instantiated via `factory`:

```pycon

>>> from datetime import date
>>> from coola.factory import resolve_object
>>> resolve_object(date(2020, 1, 1), cls=date)
datetime.date(2020, 1, 1)
>>> resolve_object({"_target_": "datetime.date", "year": 2020, "month": 1, "day": 1}, cls=date)
datetime.date(2020, 1, 1)

```

!!! warning
Any `dict` (including `dict` subclasses such as `Counter` or `OrderedDict`) is always
treated as a factory configuration, even when it is already a valid instance of the
expected class. Do not use `resolve_object` to resolve objects whose expected type is
itself a `dict` subclass.

## Security: restricting the targets

!!! danger "Only resolve configurations that you trust"

    `import_object`, `factory` and `resolve_object` import and call whatever the `_target_` names
    (for example `"os.system"`). Never resolve a configuration that comes from untrusted input.

If a configuration cannot be fully trusted (for example, a file edited by users), restrict the
modules that `_target_` may reference with `allowed_prefixes`. A target is allowed if it is equal
to a prefix, or starts with the prefix followed by a dot:

```pycon
>>> from coola.factory import resolve_object
>>> config = {"_target_": "collections.Counter", "a": 2}
>>> resolve_object(config, allowed_prefixes=["collections"])
Counter({'a': 2})
>>> resolve_object({"_target_": "os.getcwd"}, allowed_prefixes=["collections"])
Traceback (most recent call last):
    ...
ImportError: Object path 'os.getcwd' is not allowed. Allowed prefixes: ['collections']

```

`allowed_prefixes` is checked before the module is imported. The caller's list replaces the
`_allowed_prefixes_` key of the configuration, so a configuration cannot widen it. It is also
available in `resolve_loader`, `resolve_saver`, `is_object_config`, `is_loader_config` and
`is_saver_config`. For `factory`, use the `_allowed_prefixes_` key.

This is a mitigation, not a sandbox: it limits _what_ can be called, not the arguments it is called
with. Only allow objects that are safe to call with arbitrary arguments.

## Properties

| Property                  | Statement                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Explicit target           | The object to import or call is named only by the `_target_` key. The other keys are passed as keyword arguments.              |
| Fails loudly              | A target that cannot be imported raises `ImportError`, and a missing object is never replaced by a default.                    |
| Opt-in restriction        | `allowed_prefixes` is checked **before** the module is imported. Without it, any importable object can be called.              |
| Not widenable by the data | The `allowed_prefixes` argument of the caller replaces the `_allowed_prefixes_` key, so a configuration cannot widen the list. |
| Not a sandbox             | The restriction limits _what_ is called, not the arguments. Never resolve an untrusted configuration without it.               |

## Checking if a configuration targets a class

`is_object_config` checks whether the `_target_` of a configuration resolves to (or returns,
for functions with a type-hinted return value) a given class. The target is imported to check it, so
pass `allowed_prefixes` for a configuration that is not trusted. A function whose return type hint
cannot be resolved is not a configuration for the class:

```pycon

>>> from coola.factory import is_object_config
>>> from collections import Counter
>>> is_object_config({"_target_": "collections.Counter", "iterable": [1, 2, 1, 3]}, Counter)
True

```

See the [reference](../refs/factory.md) for the complete API.
