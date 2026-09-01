#!/bin/bash -eu
# Build script for ClusterFuzzLite/OSS-Fuzz Python fuzzing.
#
# Installs coola and its fuzz-relevant optional extras, then packages each
# fuzz harness under fuzz/ into a standalone executable via
# compile_python_fuzzer (provided by the oss-fuzz-base/base-builder-python
# image).

pip3 install .[numpy,torch]

for fuzzer in $(find "$SRC"/coola/fuzz -name 'fuzz_*.py'); do
  compile_python_fuzzer "$fuzzer"
done
