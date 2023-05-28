#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
##############################
# cuCollections Style Tester #
##############################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH
# LC_ALL=C.UTF-8
# LANG=C.UTF-8

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`pre-commit run clang-format --all-files 2>&1`
CLANG_FORMAT_RETVAL=$?

# Run doxygen check
DOXYGEN_CHECK=`.github/ci/pre-commit/doxygen.sh`
DOXYGEN_CHECK_RETVAL=$?

echo -e "$DOXYGEN_CHECK"

RETVALS=(
  $CLANG_FORMAT_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
