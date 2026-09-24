#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage()
{
  cat <<'EOF'
Usage:
  artifact/run_in_docker.sh prepare <CUDA architecture>
  artifact/run_in_docker.sh <smoke|full> [NVBench arguments...]

Modes:
  prepare  Download dependencies and build without requiring a GPU.
  smoke  Build the artifact and run a reduced functional evaluation.
  full   Run the paper-scale benchmark matrix.
EOF
}

select_host_device()
{
  if [[ -n "${GPU_DEVICE:-}" ]]; then
    printf '%s\n' "${GPU_DEVICE}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to select a GPU. Set GPU_DEVICE explicitly." >&2
    return 1
  fi

  local gpu_uuid
  gpu_uuid="$(
    nvidia-smi --query-gpu=uuid --format=csv,noheader |
      sed -n '1p' |
      tr -d '\r'
  )"
  if [[ -z "${gpu_uuid}" ]]; then
    echo "No GPU is visible. Set GPU_DEVICE explicitly if one is available." >&2
    return 1
  fi

  printf '%s\n' "${gpu_uuid}"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mode="$1"
shift

case "${mode}" in
  prepare)
    if [[ $# -lt 1 ]]; then
      echo "prepare requires a CUDA architecture, for example: prepare 100" >&2
      exit 1
    fi
    cuda_architectures="$1"
    shift
    output_name=prepare
    build_only=1
    docker_gpu_args=()
    gups_gpu_arch="${GUPS_GPU_ARCH:-${cuda_architectures}}"
    num_inputs=1000000
    filter_sizes=32
    default_nvbench_args=()
    ;;
  smoke)
    cuda_architectures="${CUDA_ARCHITECTURES:-native}"
    num_inputs=1000000
    filter_sizes=32
    output_name=smoke
    build_only=0
    host_device="$(select_host_device)"
    docker_gpu_args=(--gpus "device=${host_device}")
    gups_gpu_arch="${GUPS_GPU_ARCH:-}"
    gups_repeats=1
    default_nvbench_args=(--profile)
    ;;
  full)
    cuda_architectures="${CUDA_ARCHITECTURES:-native}"
    num_inputs=1000000000
    filter_sizes=32,1024
    output_name=full
    build_only=0
    host_device="$(select_host_device)"
    docker_gpu_args=(--gpus "device=${host_device}")
    gups_gpu_arch="${GUPS_GPU_ARCH:-}"
    gups_repeats=5
    default_nvbench_args=()
    ;;
  *)
    usage
    exit 1
    ;;
esac

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_workspace="/workspace/cuCollections"
source_commit="$(git -C "${root_dir}" rev-parse HEAD 2>/dev/null || true)"
image="rapidsai/devcontainers:26.10-cpp-gcc14-cuda13.3-ubuntu24.04@sha256:cc412951e7384e28a1eae61f887b5b935a2a28fe1b28852241bbe898454b1a1f"
telemetry_interval="${GPU_TELEMETRY_INTERVAL:-5}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required." >&2
  exit 1
fi

if [[ "${build_only}" == "0" ]]; then
  echo "Exposing host GPU ${host_device} as container device 0"
fi

mount_args=(-v "${root_dir}:${container_workspace}")
if git_common_dir="$(git -C "${root_dir}" rev-parse --git-common-dir 2>/dev/null)"; then
  if [[ "${git_common_dir}" != /* ]]; then
    git_common_dir="$(cd "${root_dir}/${git_common_dir}" && pwd)"
  fi
  if [[ "${git_common_dir}" != "${root_dir}"/* ]]; then
    mount_args+=(-v "${git_common_dir}:${git_common_dir}")
  fi
fi

nvbench_args=("${default_nvbench_args[@]}" "$@")

docker run --rm \
  "${docker_gpu_args[@]}" \
  --user "$(id -u):$(id -g)" \
  --workdir "${container_workspace}" \
  --env HOME=/tmp \
  --env CC=gcc \
  --env CXX=g++ \
  --env CUDAHOSTCXX=g++ \
  --env NVIDIA_DISABLE_REQUIRE=true \
  --env "BUILD_DIR=${container_workspace}/build/ia3-artifact-docker" \
  --env "OUTPUT_DIR=${container_workspace}/build/ia3-artifact-results/${output_name}" \
  --env "NUM_INPUTS=${num_inputs}" \
  --env "FILTER_SIZES=${filter_sizes}" \
  --env "CUDA_ARCHITECTURES=${cuda_architectures}" \
  --env DEVICE=0 \
  --env GUPS_LOGN=27 \
  --env "GUPS_REPEATS=${gups_repeats:-1}" \
  --env "GUPS_GPU_ARCH=${gups_gpu_arch}" \
  --env "GPU_TELEMETRY_INTERVAL=${telemetry_interval}" \
  --env "ARTIFACT_BUILD_ONLY=${build_only}" \
  --env "SOURCE_COMMIT=${source_commit}" \
  "${mount_args[@]}" \
  "${image}" \
  "${container_workspace}/artifact/run_benchmarks.sh" \
  "${nvbench_args[@]}"
