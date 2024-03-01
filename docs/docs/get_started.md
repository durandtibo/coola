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

It is recommended to create a Python 3.8+ virtual environment. This step is optional so you
can skip it. To create a virtual environment, you can use the following command:

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

The last version of `coola` is tested for the following package versions:

| package  | tested versions    |
|----------|--------------------|
| `jax`    | `>=0.4,<0.5`       |
| `numpy`  | `>=1.21,<1.27`     |
| `pandas` | `>=1.3,<2.3`       |
| `polars` | `>=0.18.3,<0.21`   |
| `torch`  | `>=1.10,<2.3`      |
| `xarray` | `>=2023.2,<2024.3` |

- More information can be found in the CI workflow configuration.
- `coola` relies on the semantic versioning (SemVer) of the packages to test the range of versions.
- `xarray` uses [calendar versioning (CalVer)](https://calver.org/) and `coola` is tested on the
  versions for the last year.
