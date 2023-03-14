#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.
##############################################i###
# cuCollections GPU build and test script for CI #
##################################################
set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Install Dependencies"
. /opt/conda/etc/profile.d/conda.sh
conda create -y -n cuda -c nvidia -c conda-forge "cudatoolkit=${CUDA_VER}" "cmake>=3.23.1"
conda activate cuda

gpuci_logger "Check versions"
python --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build from Source
################################################################################

gpuci_logger "Build Tests/Examples"
cd ${WORKSPACE}
mkdir -p build
cd build
cmake ..
make

################################################################################
# TEST - Run Tests
################################################################################

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    gpuci_logger "Check GPU usage"
    nvidia-smi
    cd ${WORKSPACE}/build/tests
    ./STATIC_SET_TEST
    ctest .

    # This block may provide more verbose testing output since each test is ran individually
    #cd ${WORKSPACE}/build/tests
    #for gt in "$WORKSPACE/build/tests"* ; do
    #    test_name=$(basename ${gt})
    #    echo "Running $test_name"
    #    ${gt}
    #done
fi
