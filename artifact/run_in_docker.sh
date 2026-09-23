#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage()
{
  cat <<'EOF'
Usage: artifact/run_in_docker.sh <smoke|full> [NVBench arguments...]

Modes:
  smoke  Build the artifact and run a reduced functional evaluation.
  full   Run the paper-scale benchmark matrix.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

mode="$1"
shift

case "${mode}" in
  smoke)
    num_inputs=1000000
    filter_sizes=32
    output_name=smoke
    default_nvbench_args=(--profile)
    ;;
  full)
    num_inputs=1000000000
    filter_sizes=32,1024
    output_name=full
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
device="${DEVICE:-0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required." >&2
  exit 1
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
  --gpus all \
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
  --env "DEVICE=${device}" \
  --env "SOURCE_COMMIT=${source_commit}" \
  "${mount_args[@]}" \
  "${image}" \
  "${container_workspace}/artifact/run_benchmarks.sh" \
  "${nvbench_args[@]}"
