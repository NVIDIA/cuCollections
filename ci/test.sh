#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
##################################
#   cuCollections test for CI    #
##################################

#set -xeuo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

source ./build.sh "$@"

ctest --test-dir ${BUILD_DIR}/tests --output-on-failure --timeout 15

echo "Test complete"