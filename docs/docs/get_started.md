# Get Started

We highly recommend installing
`coola` in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to avoid dependency conflicts.

## Using uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package installer and resolver:

```shell
uv pip install coola
```

**Install with several optional dependencies:**

```shell
uv pip install coola[all,jax,numpy,packaging,pandas,polars,pyarrow,pydantic,scipy,torch,xarray,testing,colorlog,rich]  # "all" installs everything
```

**Install with specific optional dependencies:**

```shell
uv pip install coola[numpy,torch]  # with NumPy and PyTorch
```

## First steps

After the installation, compare, hash and summarize a nested object:

```pycon
>>> from coola.equality import assert_objects_equal, compare, objects_are_equal
>>> from coola.hashing import hash_object
>>> from coola.summary import summarize
>>> actual = {"scores": [95, 87]}
>>> expected = {"scores": [95, 88]}
>>> objects_are_equal(actual, expected)
False
>>> compare(actual, expected).path
('scores', 1)
>>> assert_objects_equal(actual, expected)  # in a test: the AssertionError shows the path
Traceback (most recent call last):
    ...
AssertionError: objects are not equal at actual['scores'][1]
numbers are different:
  actual   : 87
  expected : 88
>>> len(hash_object(actual))
64
>>> print(summarize(actual))
<class 'dict'> (length=1)
  (scores): [95, 87]

```

## Using pip

Alternatively, you can use `pip`:

```shell
pip install coola
```

**Install with several optional dependencies:**

```shell
pip install coola[all,jax,numpy,packaging,pandas,polars,pyarrow,pydantic,scipy,torch,xarray,testing,colorlog,rich]  # "all" installs everything
```

**Install with specific optional dependencies:**

```shell
pip install coola[numpy,torch]  # with NumPy and PyTorch
```

## Installing from source

To install `coola` from source, you can follow the steps below.

First, clone the git repository:

```shell
git clone git@github.com:durandtibo/coola.git
cd coola
```

**Note**: `coola` requires Python 3.11 or higher.

It is recommended to create a virtual environment (this step is optional).
To create a virtual environment, you can use the following command:

```shell
make setup-venv
```

This command automatically creates a virtual environment using [`uv`](https://docs.astral.sh/uv/).
When the virtual environment is created, you can activate it with the following command:

```shell
source .venv/bin/activate
```

Then, you should install the required packages to use `coola` with the following command:

```shell
inv env.install --groups dev,docs
```

This command will install all the required packages. You can also use this command to update the
required packages. This command will check if there is a more recent package available and will
install it. Finally, you can test the installation with the following command:

```shell
inv test.unit --cov
```
