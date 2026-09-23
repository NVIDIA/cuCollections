#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root_dir}/build/ia3-artifact}"
output_dir="${OUTPUT_DIR:-${root_dir}/build/ia3-artifact-results}"
device="${DEVICE:-0}"
logn="${GUPS_LOGN:-27}"
repeats="${GUPS_REPEATS:-5}"
accesses_per_element="${GUPS_ACCESSES_PER_ELEMENT:-32}"
jobs="${JOBS:-$(nproc)}"

gups_commit="3350d216083a902ccbf5b31665e3b82096a75b55"
gups_archive_sha256="9372d0c1a6302da7f0aea13bd83cf93ee5c3d6d19eb9c1a1930c9442d790258f"
gups_archive="${build_dir}/nvidia-code-samples-${gups_commit}.tar.gz"

compute_capability="$(
  nvidia-smi \
    --id="${device}" \
    --query-gpu=compute_cap \
    --format=csv,noheader |
    head -n 1 |
    tr -d '.'
)"
gpu_arch="${GUPS_GPU_ARCH:-${compute_capability}}"
gups_root="${build_dir}/nvidia-gups-sm${gpu_arch}"
gups_source="${gups_root}/code-samples-${gups_commit}/posts/gups"

mkdir -p "${build_dir}" "${output_dir}"

if [[ ! -f "${gups_archive}" ]]; then
  curl --location --fail --silent --show-error \
    "https://github.com/NVIDIA-developer-blog/code-samples/archive/${gups_commit}.tar.gz" \
    --output "${gups_archive}"
fi

echo "${gups_archive_sha256}  ${gups_archive}" | sha256sum --check

if [[ ! -d "${gups_source}" ]]; then
  mkdir -p "${gups_root}"
  tar --extract --gzip --file "${gups_archive}" --directory "${gups_root}"
fi

make --directory "${gups_source}" GPU_ARCH="${gpu_arch}" --jobs="${jobs}"

run_test()
{
  local name="$1"
  local test_id="$2"
  local raw_output="${output_dir}/gups_${name}.txt"

  "${gups_source}/gups" \
    -n "${logn}" \
    -t "${test_id}" \
    -d "${device}" \
    -o 100 \
    -r "${repeats}" \
    -a "${accesses_per_element}" |
    tee "${raw_output}" >&2

  sed -n 's/^Result .* = //p' "${raw_output}" | tail -n 1
}

read_gups="$(run_test read 1)"
write_gups="$(run_test write 2)"
table_bytes="$(( (1 << logn) * 8 ))"

cat >"${output_dir}/gups.csv" <<EOF
device,table_log2,table_bytes,accesses_per_element,repeats,read_gups,write_gups,source_commit
${device},${logn},${table_bytes},${accesses_per_element},${repeats},${read_gups},${write_gups},${gups_commit}
EOF
