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

## Properties

These rules hold for every function of the package. Each one is enforced in a single place, not by
the discipline of each entry point.

| Property                                    | Statement                                                                                                                                                                                                                  |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Explicit target                             | The object to import or call is named only by the `_target_` string. Every other key is an argument, passed unchanged, by keyword. The reserved keys are `_target_`, `_init_` and `_allowed_prefixes_`.                    |
| No collision                                | No parameter of the API can shadow an argument of the target: the target receives every key that is not reserved, whatever its name (`obj`, `cls`, `config`, ...).                                                         |
| Top level only                              | `_target_` is resolved once per call, at the top level. A nested dictionary is data and is passed unchanged.                                                                                                               |
| Import once, call once                      | The target is imported once per call and called once. Nothing else is imported or called.                                                                                                                                  |
| Fails loudly                                | A target that does not exist raises `ImportError` with one message, whichever function is used. A wrong type raises `TypeError`. A default is never substituted for a missing object.                                      |
| Check before call                           | When `cls` is given, a target that is known not to build a `cls` is refused **before** it is called. The result is checked again after the call. With `strict=True`, a target whose result type is unknown is refused too. |
| Opt-in restriction                          | `allowed_prefixes` is checked against the string **before** the module is imported. An empty list allows nothing. Without it, any importable object can be called.                                                         |
| Policy comes from code                      | The allow-list is an argument of the caller. `resolve_object` ignores the `_allowed_prefixes_` key of a configuration, so the data cannot set, widen or narrow the list.                                                   |
| No nested configuration under an allow-list | With `allowed_prefixes`, an argument that is itself a configuration (a `dict` with `_target_`) is refused. Resolve it first, with the same list.                                                                           |
| Public constructors only                    | `_init_` is `"__init__"` (default) or the name of a public class method or static method that returns an instance of the class. A name that starts with `_` is refused.                                                    |
| Predicates do not import                    | `is_object_config` imports nothing by default. `allowed_prefixes` or `import_target=True` opt in. A malformed configuration is `False`; a violation of the allow-list raises.                                              |
| Pure, stateless                             | The configuration is never modified. The package keeps no registry, cache or lock, so it is safe to call from any thread and after a `fork`.                                                                               |
| Lazy                                        | Nothing is imported at import time except the standard library. A target is imported when it is named.                                                                                                                     |
| Keyword-only options                        | `allowed_prefixes`, `import_target`, `strict`, `_init_` and `_allowed_prefixes_` are keyword-only.                                                                                                                         |

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

By default, `factory` calls the class, which runs `__new__` and `__init__`. The special
`OBJECT_INIT` key (`"_init_"`) names a public class method or static method to call instead,
for example an alternative constructor. The method must return an instance of the class. A name
that starts with `_` (such as `"__new__"`) is refused. `_init_` is ignored when the target is a
function:

```pycon

>>> from coola.factory import factory, OBJECT_INIT
>>> factory("collections.OrderedDict", ["a", "b"], _init_="fromkeys")
OrderedDict({'a': None, 'b': None})
>>> factory("collections.OrderedDict", a=1)
OrderedDict({'a': 1})

```

## Resolving an object or a configuration

`resolve_object` is convenient in code that accepts either an already-built instance or a
configuration dictionary. If the input is already an instance of the expected type, it is
returned unchanged; if it is a `dict` with a `_target_` key, it is built:

```pycon

>>> from datetime import date
>>> from coola.factory import resolve_object
>>> resolve_object(date(2020, 1, 1), cls=date)
datetime.date(2020, 1, 1)
>>> resolve_object({"_target_": "datetime.date", "year": 2020, "month": 1, "day": 1}, cls=date)
datetime.date(2020, 1, 1)

```

A `dict` that has the key `_target_` is always a configuration, whatever `cls` is: the key is
reserved, so such a `dict` cannot be passed through `resolve_object` as data. Any other object,
including a `dict` without `_target_`, is returned as is if it is an instance of `cls`.

### Checking the type of the result

When `cls` is given (a class or a tuple of classes), `resolve_object` checks the type of what the
target builds **before** it calls the target, from the types only: a class must be a subclass of
`cls`, and a function is checked with its return type hint. A hint that is a union (`Dog | None`) or
a generic (`list[int]`) is read member by member, and the target is accepted if one of the members
is a subclass of `cls`. A target is refused before the call, with a `TypeError`, when none of them is:

```pycon

>>> from collections import OrderedDict
>>> from coola.factory import resolve_object
>>> resolve_object({"_target_": "collections.OrderedDict"}, cls=OrderedDict)
OrderedDict()
>>> resolve_object({"_target_": "collections.OrderedDict"}, cls=list)
Traceback (most recent call last):
    ...
TypeError: Received object is not a list instance (the target 'collections.OrderedDict' does not build one: it was not called)

```

The type of the result is unknown beforehand for a built-in function, for a function without a
return hint, and for a hint that cannot be resolved (a forward reference to a missing name, `Any`,
a union with a member that is not a class). Such a target is called, and its result is checked
afterwards. With `strict=True`, it is refused before the call instead. The result is always checked
again after the call, so a function that returns the wrong type is detected, but its side effects
have already happened. Calling a target is calling user code.

An abstract class is refused with a `TypeError`, and so is a target that is not callable.

## Nested configurations

`_target_` is resolved at the top level only. A nested dictionary that has a `_target_` key is
passed to the target as a plain `dict`, it is **not** instantiated (unlike Hydra, which
instantiates the nested configurations):

```pycon

>>> from coola.factory import resolve_object
>>> config = {
...     "_target_": "collections.OrderedDict",
...     "when": {"_target_": "datetime.date", "year": 2020, "month": 1, "day": 1},
... }
>>> resolve_object(config)
OrderedDict({'when': {'_target_': 'datetime.date', 'year': 2020, 'month': 1, 'day': 1}})

```

This is deliberate: `allowed_prefixes` (and `_allowed_prefixes_`) applies to the one `_target_` of
the call, so a configuration cannot make `coola` import more modules by nesting targets. With an
allow-list, a nested configuration (a top-level argument that is a `dict` with `_target_`) is
refused with a `ValueError`, because the target could resolve it without the list. To build the
nested object, resolve it yourself, with the same restrictions, and pass the result:

```pycon

>>> from datetime import date
>>> when = resolve_object(
...     {"_target_": "datetime.date", "year": 2020, "month": 1, "day": 1},
...     cls=date,
...     allowed_prefixes=["datetime"],
... )
>>> resolve_object({"_target_": "collections.OrderedDict", "when": when})
OrderedDict({'when': datetime.date(2020, 1, 1)})

```

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

`allowed_prefixes` is checked before the module is imported. `resolve_object` ignores the
`_allowed_prefixes_` key of a configuration: the list always comes from the code of the caller, so
the data cannot set, widen or narrow it. It is also
available in `resolve_loader`, `resolve_saver`, `is_object_config`, `is_loader_config` and
`is_saver_config`. For `factory`, use the `_allowed_prefixes_` key.

This is a mitigation, not a sandbox: it limits _what_ can be called, not the arguments it is called
with. Only allow objects that are safe to call with arbitrary arguments.

!!! danger "Never allow a prefix that contains a function that resolves its arguments"

    A prefix that contains a function that imports or resolves objects from its arguments (any
    `resolve_*`, `import_object`, `importlib`, `pickle`, `eval`, `os.system`, ...) makes the list
    worthless. In particular, `"coola"` and `"coola.factory"` must never be allowed prefixes.
    For example, `coola.io.resolve_loader` resolves its `loader` argument without the allow-list:

    ```python
    factory(
        "coola.io.resolve_loader",
        loader={"_target_": "os.getcwd"},
        _allowed_prefixes_=["coola.io"],
    )  # calls os.getcwd
    ```

    The same holds for a path that is checked but not the module that defines the object: a
    re-export (`mypkg.helper` defined elsewhere) is allowed by `"mypkg"`.

    Allow the narrowest prefix, and resolve nested configurations yourself, with the same list.

## Checking if a configuration targets a class

`is_object_config` checks whether the `_target_` of a configuration resolves to (or returns,
for functions with a type-hinted return value) a given class. The module of the target is **not**
imported to answer: if it is not imported yet, the configuration is not a configuration for the
class (importing a module runs its top-level code). Pass `allowed_prefixes` (a matching target is
imported if needed) or `import_target=True` (any target is imported) to import it; do this only for
a trusted configuration. A function whose return type hint cannot be resolved is not a
configuration for the class:

```pycon

>>> from coola.factory import is_object_config
>>> from collections import Counter
>>> is_object_config({"_target_": "collections.Counter", "iterable": [1, 2, 1, 3]}, Counter)
True

```

## Limits

- **An allow-list is a filter on a path, not a sandbox.** It limits what is called, not the
  arguments, and not what an allowed function does.
- **Configurations are `dict`.** JSON and YAML load into `dict`. Convert any other mapping (for
  example a `DictConfig`) before the call.
- **No attribute paths.** `_target_` is `module.object`: `collections.OrderedDict.fromkeys` does not
  exist as a target. Use `_init_` to call a class method, and a module-level alias for a nested class.
- **No recursion, positional arguments, partial application or interpolation.** There is no
  Hydra-style `_recursive_`, `_args_`, `_partial_` or `${...}`: a configuration is keyword-only
  (`factory` accepts `*args` for code that has them), a nested object is resolved by the caller, and
  `functools.partial` can be applied to the result.
- **No registry of aliases.** To look up a class by a short key, use [`coola.registry`](registry.md).
- **Two `import_object`.** `coola.factory.import_object("collections.Counter")` is the user-facing
  function (dotted path, allow-list). `coola.utils.imports.import_object("collections:Counter")`
  (colon path) is the internal function of the lazy registries.

See the [reference](../refs/factory.md) for the complete API.
