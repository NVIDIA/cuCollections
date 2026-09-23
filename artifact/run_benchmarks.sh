#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${root_dir}/build/ia3-artifact}"
output_dir="${OUTPUT_DIR:-${root_dir}/build/ia3-artifact-results}"
num_inputs="${NUM_INPUTS:-1000000000}"
filter_sizes="${FILTER_SIZES:-32,1024}"
cuda_architectures="${CUDA_ARCHITECTURES:-native}"
jobs="${JOBS:-$(nproc)}"
device="${DEVICE:-0}"
build_only="${ARTIFACT_BUILD_ONLY:-0}"
source_commit="${SOURCE_COMMIT:-}"
nvbench_args=("$@")

cmake -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
  -DGPU_ARCHS="${cuda_architectures}" \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DCUCO_DOWNLOAD_ROARING_TESTDATA=OFF

cmake --build "${build_dir}" \
  --target \
    BLOOM_FILTER_SBF_BENCH \
    BLOOM_FILTER_CSBF_BENCH \
    WARPCORE_BLOOM_FILTER_BENCH \
    BLOOM_FILTER_CBF_BENCH \
  -j "${jobs}"

if [[ "${build_only}" == "1" ]]; then
  GUPS_BUILD_ONLY=1 "${root_dir}/artifact/run_gups.sh"
  echo "Artifact dependencies and benchmark binaries are ready in ${build_dir}"
  exit 0
fi

mkdir -p "${output_dir}"
rm -f \
  "${output_dir}/gpu_sbf.json" \
  "${output_dir}/gpu_sbf.csv" \
  "${output_dir}/gpu_csbf.json" \
  "${output_dir}/gpu_csbf.csv" \
  "${output_dir}/warpcore.json" \
  "${output_dir}/warpcore.csv" \
  "${output_dir}/cbf.json" \
  "${output_dir}/cbf.csv" \
  "${output_dir}/gups.csv" \
  "${output_dir}/gups_read.txt" \
  "${output_dir}/gups_write.txt" \
  "${output_dir}/metadata.json" \
  "${output_dir}/normalized_results.csv" \
  "${output_dir}/best_results.csv" \
  "${output_dir}/sol_efficiency.csv"

run_benchmark()
{
  local name="$1"
  local executable="$2"
  shift 2

  echo "Running ${name}"
  "${build_dir}/benchmarks/${executable}" \
    "$@" \
    --json "${output_dir}/${name}.json" \
    --csv "${output_dir}/${name}.csv" \
    --quiet \
    "${nvbench_args[@]}"
}

common_axes=(
  --devices "${device}"
  --axis "NumInputs=${num_inputs}"
  --axis "FilterSizeMB=[${filter_sizes}]"
)

run_benchmark \
  "gpu_sbf" \
  "BLOOM_FILTER_SBF_BENCH" \
  "${common_axes[@]}"

run_benchmark \
  "gpu_csbf" \
  "BLOOM_FILTER_CSBF_BENCH" \
  "${common_axes[@]}"

run_benchmark \
  "warpcore" \
  "WARPCORE_BLOOM_FILTER_BENCH" \
  "${common_axes[@]}"

run_benchmark \
  "cbf" \
  "BLOOM_FILTER_CBF_BENCH" \
  "${common_axes[@]}"

"${root_dir}/artifact/run_gups.sh"

python3 - \
  "${root_dir}" \
  "${output_dir}" \
  "${num_inputs}" \
  "${filter_sizes}" \
  "${cuda_architectures}" \
  "${nvbench_args[*]}" \
  "${device}" \
  "${source_commit}" <<'PY'
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])


def command(*args):
    try:
        return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


metadata = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": command("git", "rev-parse", "HEAD") or sys.argv[8] or "unknown",
    "num_inputs": int(sys.argv[3]),
    "filter_sizes_mb": [int(value) for value in sys.argv[4].split(",")],
    "cuda_architectures": sys.argv[5],
    "nvbench_arguments": sys.argv[6],
    "device": int(sys.argv[7]),
    "platform": platform.platform(),
    "cuda_compiler": command("nvcc", "--version"),
    "host_compiler": command("c++", "--version"),
    "gpu": command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ),
}

(output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY

python3 "${root_dir}/artifact/summarize_results.py" "${output_dir}"

echo "Results written to ${output_dir}"
