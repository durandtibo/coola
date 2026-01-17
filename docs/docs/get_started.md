# Get Started

We highly recommend installing
`coola` in a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to avoid dependency conflicts.

## Using uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package installer and resolver:

```shell
uv pip install coola
```

**Install with all optional dependencies:**

```shell
uv pip install coola[all]
```

**Install with specific optional dependencies:**

```shell
uv pip install coola[numpy,torch]  # with NumPy and PyTorch
```

## Using pip

Alternatively, you can use `pip`:

```shell
pip install coola
```

**Install with all optional dependencies:**

```shell
pip install coola[all]
```

**Install with specific optional dependencies:**

```shell
pip install coola[numpy,torch]  # with NumPy and PyTorch
```

## Requirements

- **Python**: 3.10 or higher
- **Core dependencies**: None (fully optional dependencies)

**Optional dependencies** (install with `coola[all]`):
[JAX](https://jax.readthedocs.io/) •
[NumPy](https://numpy.org/) •
[pandas](https://pandas.pydata.org/) •
[polars](https://www.pola.rs/) •
[PyArrow](https://arrow.apache.org/docs/python/) •
[PyTorch](https://pytorch.org/) •
[xarray](https://docs.xarray.dev/)

## Compatibility Matrix

| `coola`  | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `packaging`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `pyarrow`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python`       |
|----------|-------------------|---------------------|-------------------------|----------------------|----------------------|-----------------------|---------------------|----------------------|----------------|
| `main`   | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10`       |
| `0.11.1` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10`       |
| `0.11.0` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10`       |
| `0.10.0` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10`       |

<sup>*</sup> indicates an optional dependency

<details>
    <summary>older versions</summary>

| `coola`  | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `packaging`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `pyarrow`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python`      |
|----------|-------------------|---------------------|-------------------------|----------------------|----------------------|-----------------------|---------------------|----------------------|---------------|
| `0.9.1`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0,<26.0`          | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10,<3.15` |
| `0.9.0`  | `>=0.4.6,<1.0`    | `>=1.24,<3.0`       | `>=22.0,<26.0`          | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<20.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.9,<3.14`  |
| `0.8.7`  | `>=0.4.6,<1.0`    | `>=1.22,<3.0`       | `>=21.0,<26.0`          | `>=1.5,<3.0`         | `>=1.0,<2.0`         | `>=10.0,<20.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |

</details>

For the most up-to-date compatibility information, see:

- The [GitHub README compatibility table](https://github.com/durandtibo/coola#installation)
- The [CI workflow configuration](https://github.com/durandtibo/coola/tree/main/.github/workflows/)
- The [pyproject.toml file](https://github.com/durandtibo/coola/blob/main/pyproject.toml)

## Notes

- `coola` relies on semantic versioning (SemVer) for most packages.
- `xarray` uses [calendar versioning (CalVer)](https://calver.org/).
- Version constraints are designed to be flexible while ensuring compatibility.
