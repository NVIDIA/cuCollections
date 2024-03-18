/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuco/distinct_count_estimator.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>

/**
 * @file device_ref_example.cu
 * @brief Demonstrates usage of `cuco::distinct_count_estimator` device-side APIs.
 *
 * This example demonstrates how the non-owning reference type `cuco::distinct_count_estimator_ref`
 * can be used to implement a custom kernel that fuses the cardinality estimation step with any
 * other workload that traverses the input data.
 */

template <class RefType, class InputIt>
__global__ void piggyback_kernel(RefType ref, InputIt first, std::size_t n)
{
  // Transform the reference type (with device scope) to a reference type with block scope
  using local_ref_type = typename RefType::with_scope<cuda::thread_scope_block>;

  // Shared memory storage for the block-local estimator
  extern __shared__ std::byte local_sketch[];

  auto const loop_stride = gridDim.x * blockDim.x;
  auto idx               = blockDim.x * blockIdx.x + threadIdx.x;
  auto const block       = cooperative_groups::this_thread_block();

  // Create the local estimator with the shared memory storage
  local_ref_type local_ref(cuda::std::span{local_sketch, ref.sketch_bytes()});

  // Initialize the local estimator
  local_ref.clear(block);
  block.sync();

  while (idx < n) {
    auto const& item = *(first + idx);

    // Add each item to the local estimator
    local_ref.add(item);

    /*
    Here we can add some custom workload that takes the input `item`.

    The idea is that cardinality estimation can be fused/piggy-backed with any other workload that
    traverses the data. Since `local_ref.add` can run close to the SOL of the DRAM bandwidth, we get
    the estimate "for free" while performing other computations over the data.
    */

    idx += loop_stride;
  }
  block.sync();

  // We can also compute the local estimate on the device
  auto const local_estimate = local_ref.estimate(block);
  if (block.thread_rank() == 0) {
    // The local estimate should approximately be `num_items`/`gridDim.x`
    printf("Estimate for block %d = %llu\n", blockIdx.x, local_estimate);
  }

  // In the end, we merge the shared memory estimator into the global estimator which gives us the
  // final result
  ref.merge(block, local_ref);
}

int main(void)
{
  using T                         = int;
  constexpr std::size_t num_items = 1ull << 28;  // 1GB

  thrust::device_vector<T> items(num_items);

  // Generate `num_items` distinct items
  thrust::sequence(items.begin(), items.end(), 0);

  // Initialize the estimator
  cuco::distinct_count_estimator<T> estimator;

  // Add all items to the estimator
  estimator.add(items.begin(), items.end());

  // Calculate the cardinality estimate from the bulk operation
  std::size_t const estimated_cardinality_bulk = estimator.estimate();

  // Clear the estimator so it can be reused
  estimator.clear();

  // Number of dynamic shared memory bytes required to store a CTA-local sketch
  auto const sketch_bytes = estimator.sketch_bytes();

  // Call the custom kernel and pass a non-owning reference to the estimator to the GPU
  piggyback_kernel<<<10, 512, sketch_bytes>>>(estimator.ref(), items.begin(), num_items);

  // Calculate the cardinality estimate from the custom kernel
  std::size_t const estimated_cardinality_custom = estimator.estimate();

  if (estimated_cardinality_bulk == estimated_cardinality_custom) {
    std::cout << "Success! Cardinality estimates are identical" << std::endl;
  }

  return 0;
}