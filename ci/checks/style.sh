#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
DOXYGEN_CHECK=`ci/checks/doxygen.sh`
DOXYGEN_CHECK_RETVAL=$?
echo -e "$DOXYGEN_CHECK_RETVAL"

RETVALS=(
  $CLANG_FORMAT_RETVAL $DOXYGEN_CHECK_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
