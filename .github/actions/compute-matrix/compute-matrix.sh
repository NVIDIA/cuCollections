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

set -euo pipefail

# Check for the correct number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 MATRIX_FILE MATRIX_QUERY"
    echo "MATRIX_FILE: The path to the matrix file."
    echo "MATRIX_QUERY: The jq query used to specify the desired matrix. e.g., '.pull-request.nvcc'"
    exit 1
fi

# Get realpath before changing directory
MATRIX_FILE=$(realpath "$1")
MATRIX_QUERY="$2"

# Ensure the script is being executed in its containing directory
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )";

echo "Input matrix file:" >&2
cat "$MATRIX_FILE" >&2
echo "Query: $MATRIX_QUERY" >&2
echo $(yq -o=json "$MATRIX_FILE" | jq -c -r "$MATRIX_QUERY | map(. as \$o | {std: .std[]} + del(\$o.std))")