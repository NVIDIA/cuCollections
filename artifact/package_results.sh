#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: artifact/package_results.sh <result-directory> [archive.tar.gz]" >&2
  exit 1
fi

if [[ ! -d "$1" ]]; then
  echo "Result directory does not exist: $1" >&2
  exit 1
fi

results_dir="$(realpath "$1")"
archive="$(realpath -m "${2:-${results_dir}.tar.gz}")"

if [[ "${archive}" == "${results_dir}/"* ]]; then
  echo "Archive must be written outside the result directory." >&2
  exit 1
fi

required_files=(
  sbf.json
  sbf.csv
  csbf.json
  csbf.csv
  warpcore.json
  warpcore.csv
  cbf.json
  cbf.csv
  metadata.json
  normalized_results.csv
  best_results.csv
  comparisons.csv
  summary.md
  gups.csv
  gups_read.txt
  gups_write.txt
  bloom_filter_tests.log
  run.log
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${results_dir}/${file}" ]]; then
    echo "Missing required result file: ${results_dir}/${file}" >&2
    exit 1
  fi
done

optional_files=(
  gpu_state_before.csv
  gpu_state_after.csv
  gpu_telemetry.csv
)

for file in "${optional_files[@]}"; do
  if [[ ! -f "${results_dir}/${file}" ]]; then
    echo "Warning: optional diagnostic file is missing: ${results_dir}/${file}" >&2
  fi
done

mkdir -p "$(dirname "${archive}")"
tar \
  --create \
  --gzip \
  --file "${archive}" \
  --directory "$(dirname "${results_dir}")" \
  "$(basename "${results_dir}")"

archive_dir="$(dirname "${archive}")"
archive_name="$(basename "${archive}")"
(
  cd "${archive_dir}"
  sha256sum "${archive_name}" >"${archive_name}.sha256"
)

echo "Created ${archive}"
echo "Created ${archive}.sha256"
