#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eo pipefail

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

# Script defaults
BUILD_PREFIX=../build # <repo_root>/build
BUILD_INFIX=local # <repo_root>/build/local
BUILD_TYPE=Release # CMake build type
HOST_COMPILER=${CXX:-g++} # $CXX if set, otherwise `g++`
CUDA_COMPILER=${CUDACXX:-nvcc} # $CUDACXX if set, otherwise `nvcc`
CXX_STANDARD=17
PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)} # defaults to number of cores in the system
CUDA_ARCHS=native # detect system's GPU architectures

# TODO figure out how to build tests/benchmarks/examples separately

function usage {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -v/--verbose: enable shell echo for debugging"
    echo "  -prefix: Build directory prefix (Defaults to <repo_root>/build)"
    echo "  -i/-infix: Build directory infix (Defaults to local)"
    echo "  -t/-type: CMake build type (Defaults to Release)"
    echo "  -cuda: CUDA compiler (Defaults to \$CUDACXX if set, otherwise nvcc)"
    echo "  -cxx: Host compiler (Defaults to \$CXX if set, otherwise g++)"
    echo "  -arch: Target CUDA arches, e.g. \"60-real;70;80-virtual\" (Defaults to the system's native GPU archs)"
    echo "  -std: CUDA/C++ standard (Defaults to 17)"
    echo "  -p/-par: Build parallelism (Defaults to \$PARALLEL_LEVEL if set, otherwise the system's number of CPU cores)"
    echo
    echo "Examples:"
    echo "  $ PARALLEL_LEVEL=8 CXX=g++-9 $0"
    echo "  $ $0 -cxx g++-9 -par 8 -i my_build"
    echo "  $ $0 -cxx g++-8 -std 14 -arch 80-real -v -cuda /usr/local/bin/nvcc"
    exit 1
}

# Parse options

# Copy the args into a temporary array, since we will modify them and
# the parent script may still need them.
args=("$@")
echo "Args: ${args[@]}"
while [ "${#args[@]}" -ne 0 ]; do
    case "${args[0]}" in
    -v | --verbose) VERBOSE=1; args=("${args[@]:1}");;
    -prefix) BUILD_PREFIX="${args[1]}"; args=("${args[@]:2}");;
    -i | -infix) BUILD_INFIX="${args[1]}"; args=("${args[@]:2}");;
    -t | -type) BUILD_TYPE="${args[1]}"; args=("${args[@]:2}");;
    -cuda) CUDA_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -cxx)  HOST_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -arch) CUDA_ARCHS="${args[1]}";    args=("${args[@]:2}");;
    -std)  CXX_STANDARD="${args[1]}";  args=("${args[@]:2}");;
    -p | -par)  PARALLEL_LEVEL="${args[1]}";  args=("${args[@]:2}");;
    -h | -help | --help) usage ;;
    *) echo "Unrecognized option: ${args[0]}"; usage ;;
    esac
done

# Convert to full paths:
HOST_COMPILER=$(which ${HOST_COMPILER})
CUDA_COMPILER=$(which ${CUDA_COMPILER})
# Make CUDA arch list compatible with cmake
CUDA_ARCHS=$(echo "$CUDA_ARCHS" | tr ' ,' ';;')

if [ $VERBOSE ]; then
    set -x
fi

# Begin processing unsets after option parsing
set -u

if [ "$BUILD_INFIX" = "latest" ] || [ -z "$BUILD_INFIX" ]; then
    echo "Error: BUILD_INFIX cannot be 'latest'" >&2
    exit 1
fi

BUILD_DIR="$BUILD_PREFIX/$BUILD_INFIX"
export BUILD_DIR # TODO remove
mkdir -p $BUILD_DIR

# The most recent build will be symlinked to cuCollections/build/latest
rm -f $BUILD_PREFIX/latest
ln -sf $BUILD_DIR $BUILD_PREFIX/latest


# Now that BUILD_DIR exists, use readlink to canonicalize the path:
BUILD_DIR=$(readlink -f "${BUILD_DIR}")

CMAKE_OPTIONS="
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CXX_STANDARD=${CXX_STANDARD} \
    -DCMAKE_CUDA_STANDARD=${CXX_STANDARD} \
    -DCMAKE_CXX_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
    -DCMAKE_CUDA_HOST_COMPILER=${HOST_COMPILER} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
"

echo "========================================"
echo "-- START: $(date)"
echo "-- GIT_SHA: $(git rev-parse HEAD 2>/dev/null || echo 'Not a repository')"
echo "-- PWD: $(pwd)"
echo "-- BUILD_DIR: ${BUILD_DIR}"
echo "-- PARALLEL_LEVEL: ${PARALLEL_LEVEL}"
echo "-- CUDA_ARCHS: ${CUDA_ARCHS}"

# configure
cmake -S .. -B $BUILD_DIR $CMAKE_OPTIONS
echo "========================================"

if command -v sccache >/dev/null; then
    source "./sccache_stats.sh" start
fi

#build
cmake --build $BUILD_DIR --parallel $PARALLEL_LEVEL
echo "========================================"
echo "Build complete"

if command -v sccache >/dev/null; then
    source "./sccache_stats.sh" end
else
    echo "sccache stats: N/A"
fi