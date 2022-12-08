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

echo -e "$DOXYGEN_CHECK"

# Check for copyright headers in the files modified currently
COPYRIGHT=`python ci/checks/copyright.py --git-modified-only 2>&1`
COPYRIGHT_RETVAL=$?

# Output results if failure otherwise show pass
if [ "$COPYRIGHT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
  echo -e "$COPYRIGHT"
fi

RETVALS=(
  $CLANG_FORMAT_RETVAL $COPYRIGHT_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
