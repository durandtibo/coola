# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `pip` (recommended)

The following command installs the latest version of the library:

```shell
pip install coola
```

To make the package as slim as possible, only the packages required to use `coola` are installed.
It is possible to install all the optional dependencies by running the following command:

```shell
pip install 'coola[all]'
```

This command also installed NumPy and PyTorch.
It is also possible to install the optional packages manually or to select the packages to install.
In the following example, only NumPy is installed:

```shell
pip install coola numpy
```

## Installing from source

To install `coola` from source, you can follow the steps below. First, you will need to
install [`poetry`](https://python-poetry.org/docs/master/). `poetry` is used to manage and install
the dependencies.
If `poetry` is already installed on your machine, you can skip this step. There are several ways to
install `poetry` so you can use the one that you prefer. You can check the `poetry` installation by
running the following command:

```shell
poetry --version
```

Then, you can clone the git repository:

```shell
git clone git@github.com:durandtibo/coola.git
```

**Note**: `coola` requires Python 3.10 or higher.

It is recommended to create a virtual environment (this step is optional). 
To create a virtual environment, you can use the following command:

```shell
make conda
```

It automatically creates a conda virtual environment. When the virtual environment is created, you
can activate it with the following command:

```shell
conda activate coola
```

This example uses `conda` to create a virtual environment, but you can use other tools or
configurations. Then, you should install the required package to use `coola` with the following
command:

```shell
make install
```

This command will install all the required packages. You can also use this command to update the
required packages. This command will check if there is a more recent package available and will
install it. Finally, you can test the installation with the following command:

```shell
make unit-test-cov
```

## Testing

`coola` is continuously tested with the following package versions:

| package     | tested versions       |
|-------------|-----------------------|
| `jax`       | `>=0.5.0,<1.0`        |
| `numpy`     | `>=1.24,<3.0`         |
| `packaging` | `>=22.0,<26.0`        |
| `pandas`    | `>=2.0,<3.0`          |
| `polars`    | `>=1.0,<2.0`          |
| `pyarrow`   | `>=11.0,<23.0`        |
| `torch`     | `>=2.0,<3.0`          |
| `xarray`    | `>=2023.1`            |

For the most up-to-date compatibility information, see:
- The [GitHub README compatibility table](https://github.com/durandtibo/coola#installation)
- The [CI workflow configuration](https://github.com/durandtibo/coola/tree/main/.github/workflows/)
- The [pyproject.toml file](https://github.com/durandtibo/coola/blob/main/pyproject.toml)

**Note**: 
- `coola` relies on semantic versioning (SemVer) for most packages.
- `xarray` uses [calendar versioning (CalVer)](https://calver.org/).
- Version constraints are designed to be flexible while ensuring compatibility.
