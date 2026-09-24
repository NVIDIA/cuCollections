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
telemetry_interval="${GPU_TELEMETRY_INTERVAL:-5}"
source_commit="${SOURCE_COMMIT:-}"
nvbench_args=("$@")

if [[ "${build_only}" != "1" && "${telemetry_interval}" != "0" ]]; then
  if [[ ! "${telemetry_interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPU_TELEMETRY_INTERVAL must be zero or a positive integer." >&2
    exit 1
  fi
fi

if [[ "${_IA3_ARTIFACT_LOGGING_ACTIVE:-0}" != "1" ]]; then
  mkdir -p "${output_dir}"
  rm -f "${output_dir}/run.log"
  set +e
  _IA3_ARTIFACT_LOGGING_ACTIVE=1 bash "${BASH_SOURCE[0]}" "$@" 2>&1 |
    tee "${output_dir}/run.log"
  pipeline_status=("${PIPESTATUS[@]}")
  if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    exit "${pipeline_status[0]}"
  fi
  exit "${pipeline_status[1]}"
fi

mkdir -p "${output_dir}"
rm -f \
  "${output_dir}/sbf.json" \
  "${output_dir}/sbf.csv" \
  "${output_dir}/csbf.json" \
  "${output_dir}/csbf.csv" \
  "${output_dir}/warpcore.json" \
  "${output_dir}/warpcore.csv" \
  "${output_dir}/cbf.json" \
  "${output_dir}/cbf.csv" \
  "${output_dir}/gups.csv" \
  "${output_dir}/gups_read.txt" \
  "${output_dir}/gups_write.txt" \
  "${output_dir}/bloom_filter_tests.log" \
  "${output_dir}/gpu_state_before.csv" \
  "${output_dir}/gpu_state_after.csv" \
  "${output_dir}/gpu_telemetry.csv" \
  "${output_dir}/metadata.json" \
  "${output_dir}/normalized_results.csv" \
  "${output_dir}/best_results.csv" \
  "${output_dir}/sol_efficiency.csv" \
  "${output_dir}/comparisons.csv" \
  "${output_dir}/summary.md"

diagnostic_device="${NVIDIA_SMI_DEVICE:-}"
if [[ -z "${diagnostic_device}" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    diagnostic_device="${device}"
  else
    echo "GPU diagnostics disabled because CUDA_VISIBLE_DEVICES remaps device ordinals."
    echo "Set NVIDIA_SMI_DEVICE to a physical GPU index or UUID to enable them."
  fi
fi

cmake -S "${root_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
  -DGPU_ARCHS="${cuda_architectures}" \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DCUCO_DOWNLOAD_ROARING_TESTDATA=OFF

cmake --build "${build_dir}" \
  --target \
    BLOOM_FILTER_SBF_BENCH \
    BLOOM_FILTER_CSBF_BENCH \
    WARPCORE_BLOOM_FILTER_BENCH \
    BLOOM_FILTER_CBF_BENCH \
    BLOOM_FILTER_TEST \
  -j "${jobs}"

if [[ "${build_only}" == "1" ]]; then
  GUPS_BUILD_ONLY=1 "${root_dir}/artifact/run_gups.sh"
  echo "Artifact dependencies and benchmark binaries are ready in ${build_dir}"
  exit 0
fi

echo "Running Bloom filter correctness tests"
"${build_dir}/tests/BLOOM_FILTER_TEST" --reporter compact --rng-seed 12345 2>&1 |
  tee "${output_dir}/bloom_filter_tests.log"

telemetry_pid=""
gpu_snapshot_fields="timestamp,index,uuid,name,pci.bus_id,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,clocks.max.graphics,clocks.max.memory,memory.total,memory.used"
gpu_telemetry_fields="timestamp,index,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,memory.used"
stop_gpu_telemetry()
{
  if [[ -n "${telemetry_pid}" ]]; then
    kill "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
    telemetry_pid=""
  fi
}
trap stop_gpu_telemetry EXIT

if [[ -n "${diagnostic_device}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --id="${diagnostic_device}" --query-gpu="${gpu_snapshot_fields}" \
    --format=csv >"${output_dir}/gpu_state_before.csv" || true

  if [[ "${telemetry_interval}" != "0" ]]; then
    nvidia-smi --id="${diagnostic_device}" --query-gpu="${gpu_telemetry_fields}" \
      --format=csv --loop="${telemetry_interval}" >"${output_dir}/gpu_telemetry.csv" 2>&1 &
    telemetry_pid="$!"
  fi
fi

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
  "sbf" \
  "BLOOM_FILTER_SBF_BENCH" \
  "${common_axes[@]}"

run_benchmark \
  "csbf" \
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

gups_gpu_arch="${GUPS_GPU_ARCH:-}"
if [[ -z "${gups_gpu_arch}" ]]; then
  gups_gpu_arch="$(
    python3 - "${output_dir}/sbf.json" "${device}" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1]))
device_id = int(sys.argv[2])
device = next(entry for entry in document["devices"] if entry["id"] == device_id)
print(int(device["sm_version"]) // 10)
PY
  )"
fi
GUPS_GPU_ARCH="${gups_gpu_arch}" \
  NVIDIA_SMI_DEVICE="${diagnostic_device}" \
  "${root_dir}/artifact/run_gups.sh"

stop_gpu_telemetry
trap - EXIT
if [[ -n "${diagnostic_device}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --id="${diagnostic_device}" --query-gpu="${gpu_snapshot_fields}" \
    --format=csv >"${output_dir}/gpu_state_after.csv" || true
fi

python3 - \
  "${root_dir}" \
  "${output_dir}" \
  "${num_inputs}" \
  "${filter_sizes}" \
  "${cuda_architectures}" \
  "${nvbench_args[*]}" \
  "${device}" \
  "${source_commit}" \
  "${diagnostic_device}" <<'PY'
import json
import os
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


git_commit = command("git", "rev-parse", "HEAD")
git_status = command("git", "status", "--porcelain")
diagnostic_device = sys.argv[9]


def nvbench_gpu():
    try:
        document = json.loads((output / "sbf.json").read_text())
        device_id = int(sys.argv[7])
        device = next(entry for entry in document["devices"] if entry["id"] == device_id)
        memory_mib = int(device["global_memory_size"]) // (1024 * 1024)
        return f"{device['name']}, SM {int(device['sm_version']) // 10}, {memory_mib} MiB"
    except (KeyError, OSError, StopIteration, ValueError):
        return None


gpu = None
if diagnostic_device:
    gpu = command(
        "nvidia-smi",
        "--id",
        diagnostic_device,
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    )
if gpu is None:
    gpu = nvbench_gpu()

telemetry_path = output / "gpu_telemetry.csv"
try:
    telemetry_lines = telemetry_path.read_text().splitlines()
except OSError:
    telemetry_lines = []
telemetry_collected = len(telemetry_lines) > 1 and telemetry_lines[0].startswith("timestamp")

metadata = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_commit or sys.argv[8] or "unknown",
    "git_dirty": None if git_status is None else bool(git_status),
    "hostname": platform.node(),
    "num_inputs": int(sys.argv[3]),
    "filter_sizes_mb": [int(value) for value in sys.argv[4].split(",")],
    "cuda_architectures": sys.argv[5],
    "nvbench_arguments": sys.argv[6],
    "device": int(sys.argv[7]),
    "correctness_tests": "passed",
    "platform": platform.platform(),
    "cuda_compiler": command("nvcc", "--version"),
    "host_compiler": command("c++", "--version"),
    "gpu": gpu,
    "nvidia_smi_device": diagnostic_device or None,
    "gpu_telemetry_collected": telemetry_collected,
    "gpu_telemetry_interval_seconds": (
        int(os.environ.get("GPU_TELEMETRY_INTERVAL", "5")) if telemetry_collected else None
    ),
}

(output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY

python3 "${root_dir}/artifact/summarize_results.py" "${output_dir}"

echo "Results written to ${output_dir}"
