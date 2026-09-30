# Development Guide

This guide covers setting up your development environment and common development tasks.

## Prerequisites

- Python 3.10 or higher
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- Git for version control
- Basic knowledge of Python and testing

## Initial Setup

### 1. Fork and Clone

```shell
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/coola.git
cd coola
```

### 2. Set Up Virtual Environment

The project uses `uv` for dependency management. First, install `uv` if you don't have it:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and set up the virtual environment:

```shell
make setup-venv
```

This will create a virtual environment and install all dependencies, including development tools.

Activate the virtual environment:

```shell
source .venv/bin/activate
```

### 3. Install Dependencies

If you already have a virtual environment and just want to install dependencies:

```shell
# Install core dependencies
inv env.install --no-optional-deps

# Install with documentation dependencies
inv env.install --groups dev,docs
```

### 4. Set Up Pre-commit Hooks

```shell
pre-commit install
```

This will automatically run code quality checks before each commit.

## Development Workflow

### Running Tests

**Run all unit tests:**

```shell
inv test.unit
```

**Run tests with coverage:**

```shell
inv test.unit --cov
```

**Run specific test file:**

```shell
pytest tests/unit/equality/test_interface.py
```

**Run specific test:**

```shell
pytest tests/unit/equality/test_interface.py::test_objects_are_equal_false_different_type
```

### Code Quality

**Check code formatting (Ruff):**

```shell
inv format.check-python
```

**Fix code formatting:**

```shell
inv format.fix-python
```

**Run linter (Ruff):**

```shell
inv lint.check-python
```

**Check the package layering and import cycles:**

```shell
lint-imports
inv imports.check-cycles
```

**Format docstrings:**

```shell
inv format.fix-docstrings
```

**Run all pre-commit checks:**

```shell
pre-commit run --all-files
```

### Documentation

**Build documentation locally:**

```shell
mkdocs serve -f docs/mkdocs.yml
```

Then open http://127.0.0.1:8000 in your browser.

**Build documentation without serving:**

```shell
mkdocs build -f docs/mkdocs.yml
```

**Run doctests:**

```shell
inv test.doctest
```

### Type Checking

`coola` uses [ty](https://docs.astral.sh/ty/) for type checking (see the `ci-type-checking`
workflow). You can run type checking locally:

```shell
ty check
```

## Project Structure

```
coola/
├── .github/               # GitHub configuration
│   ├── workflows/        # CI/CD workflows
│   └── ISSUE_TEMPLATE/   # Issue templates
├── docs/                 # Documentation
│   ├── docs/            # Documentation source
│   └── mkdocs.yml       # MkDocs configuration
├── src/                 # Source code
│   └── coola/
│       ├── equality/             # Equality comparison
│       │   ├── tester/           # Type-specific testers and the registry
│       │   └── handler/          # Comparison handlers
│       ├── registry/, summary/, hashing/, recursive/, iterator/, nested/, ...
│       └── utils/                # Utility functions
├── tests/               # Test files
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── CONTRIBUTING.md     # Contribution guidelines
├── pyproject.toml      # Project configuration
├── uv.lock             # Locked dependencies
├── LICENSE             # License file
├── README.md           # Project README
└── SECURITY.md         # Security policy
```

## Common Development Tasks

### Adding a New Feature

1. **Create a new branch:**

   ```shell
   git checkout -b feature/my-feature
   ```

2. **Implement the feature:**
   - Write code in `src/coola/`
   - Add tests in `tests/unit/`
   - Update documentation in `docs/docs/`

3. **Run tests:**

   ```shell
   inv test.unit --cov
   ```

4. **Run code quality checks:**

   ```shell
   pre-commit run --all-files
   ```

5. **Commit changes:**

   ```shell
   git add .
   git commit -m "Add: brief description of feature"
   ```

6. **Push and create PR:**
   ```shell
   git push origin feature/my-feature
   ```

### Fixing a Bug

1. **Create a branch:**

   ```shell
   git checkout -b fix/bug-description
   ```

2. **Write a failing test** that reproduces the bug

3. **Fix the bug**

4. **Verify the test passes:**

   ```shell
   pytest tests/unit/path/to/test.py
   ```

5. **Run full test suite:**

   ```shell
   inv test.unit --cov
   ```

6. **Commit and push:**
   ```shell
   git commit -m "Fix: description of bug fix"
   git push origin fix/bug-description
   ```

### Updating Dependencies

```shell
# Update all dependencies
inv env.update

# Dependencies are managed in pyproject.toml and locked in uv.lock
# To add a new dependency, edit pyproject.toml and run:
uv lock
```

## Testing Guidelines

### Writing Good Tests

1. **Use descriptive names:**

   ```python
   def test_objects_are_equal_with_identical_dicts_returns_true(): ...


   def test_objects_are_equal_with_different_types_returns_false(): ...
   ```

2. **Test edge cases:**
   - Empty collections
   - None values
   - Large data
   - Deeply nested structures

3. **Use fixtures for common data:**

   ```python
   import pytest
   import torch
   from coola.equality import objects_are_equal


   @pytest.fixture
   def sample_tensor():
       return torch.randn(10, 10)


   def test_tensor_comparison(sample_tensor):
       result = objects_are_equal(sample_tensor, sample_tensor)
       assert result is True
   ```

4. **Test both success and failure cases:**
   ```python
   def test_success_case():
       assert objects_are_equal(obj1, obj2)


   def test_failure_case():
       assert not objects_are_equal(obj1, obj3)
   ```

## Continuous Integration

The project uses GitHub Actions for CI. Workflows are in `.github/workflows/`:

- **CI**: Runs on every push and PR
  - Linting
  - Tests
  - Coverage

- **Documentation**: Builds and deploys docs
  - Builds on every push
  - Deploys on release

- **Nightly Tests**: Tests against latest dependencies
  - Runs daily
  - Tests multiple Python versions

## Best Practices

1. **Write tests first** (TDD approach when possible)
2. **Keep PRs focused** on a single feature or fix
3. **Update documentation** for user-facing changes
4. **Run pre-commit hooks** before committing
5. **Write clear commit messages**
6. **Add docstrings** to all public APIs
7. **Keep dependencies minimal**
8. **Follow existing code style**

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] All tests pass locally
- [ ] Code coverage is maintained or improved
- [ ] Pre-commit hooks pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Code follows project style
- [ ] No unnecessary dependencies added
- [ ] Examples are provided for new features
- [ ] Edge cases are tested

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
