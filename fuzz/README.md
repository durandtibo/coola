# Fuzzing

This directory contains [Atheris](https://github.com/google/atheris) fuzz harnesses for `coola`,
run continuously on pull requests via [ClusterFuzzLite](https://google.github.io/clusterfuzzlite/)
(see `.clusterfuzzlite/` and `.github/workflows/cifuzz.yaml`).

Each `fuzz_*.py` file is a standalone entry point discovered by
`.clusterfuzzlite/build.sh` and exposes a `TestOneInput(data: bytes)` function, as required by
Atheris/libFuzzer.

## Running locally

```shell
pip install atheris
python fuzz/fuzz_objects_are_equal.py
python fuzz/fuzz_summarize.py
python fuzz/fuzz_hash_object.py
python fuzz/fuzz_flat_dict.py
python fuzz/fuzz_string_utils.py
```

Add `-runs=100000` or a corpus directory as extra CLI arguments to control the run, e.g.:

```shell
python fuzz/fuzz_objects_are_equal.py -runs=100000
```
