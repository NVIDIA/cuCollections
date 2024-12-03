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
#pragma once

#include <cuco/detail/static_map/kernels.cuh>
#include <cuco/detail/utility/cuda.cuh>

namespace cuco::detail::static_map_ns {

/**
 * @brief Dispatches to shared memory map kernel if `num_elements_per_thread > 2`, else
 * fallbacks to global memory map kernel.
 *
 * @tparam HasInit Boolean to dispatch based on init parameter
 * @tparam CGSize Number of threads in each CG
 * @tparam Allocator Allocator type used to created shared_memory map
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the `value_type` of the data structure
 * @tparam Init Type of init value convertible to payload type
 * @tparam Op Callable type used to peform `apply` operation.
 * @tparam Ref Type of non-owning device ref allowing access to storage
 *
 * @param first Beginning of the sequence of input elements
 * @param last End of the sequence of input elements
 * @param init The init value of the `op`
 * @param op Callable object to perform apply operation.
 * @param ref Non-owning container device ref used to access the slot storage
 * @param stream CUDA stream used for insert_or_apply operation
 */
template <bool HasInit,
          int32_t CGSize,
          typename Allocator,
          typename InputIt,
          typename Init,
          typename Op,
          typename Ref>
void dispatch_insert_or_apply(
  InputIt first, InputIt last, Init init, Op op, Ref ref, cuda::stream_ref stream)
{
  auto const num = cuco::detail::distance(first, last);
  if (num == 0) { return; }

  int32_t const default_grid_size = cuco::detail::grid_size(num, CGSize);

  if constexpr (CGSize == 1) {
    using shmem_size_type = int32_t;

    int32_t constexpr shmem_block_size                = 1024;
    shmem_size_type constexpr cardinality_threshold   = shmem_block_size;
    shmem_size_type constexpr shared_map_num_elements = cardinality_threshold + shmem_block_size;
    float constexpr load_factor                       = 0.7;
    shmem_size_type constexpr shared_map_size =
      static_cast<shmem_size_type>((1.0 / load_factor) * shared_map_num_elements);

    using extent_type     = cuco::extent<shmem_size_type, shared_map_size>;
    using shared_map_type = cuco::static_map<typename Ref::key_type,
                                             typename Ref::mapped_type,
                                             extent_type,
                                             cuda::thread_scope_block,
                                             typename Ref::key_equal,
                                             typename Ref::probing_scheme_type,
                                             Allocator,
                                             cuco::storage<1>>;

    using shared_map_ref_type    = typename shared_map_type::ref_type<>;
    auto constexpr bucket_extent = cuco::make_bucket_extent<shared_map_ref_type>(extent_type{});

    auto insert_or_apply_shmem_fn_ptr = insert_or_apply_shmem<HasInit,
                                                              CGSize,
                                                              shmem_block_size,
                                                              shared_map_ref_type,
                                                              InputIt,
                                                              Init,
                                                              Op,
                                                              Ref>;

    int32_t const max_op_grid_size =
      cuco::detail::max_occupancy_grid_size(shmem_block_size, insert_or_apply_shmem_fn_ptr);

    int32_t const shmem_default_grid_size =
      cuco::detail::grid_size(num, CGSize, cuco::detail::default_stride(), shmem_block_size);

    auto const shmem_grid_size         = std::min(shmem_default_grid_size, max_op_grid_size);
    auto const num_elements_per_thread = num / (shmem_grid_size * shmem_block_size);

    // use shared_memory only if each thread has atleast 3 elements to process
    if (num_elements_per_thread > 2) {
      insert_or_apply_shmem<HasInit, CGSize, shmem_block_size, shared_map_ref_type>
        <<<shmem_grid_size, shmem_block_size, 0, stream.get()>>>(
          first, num, init, op, ref, bucket_extent);
    } else {
      insert_or_apply<HasInit, CGSize, cuco::detail::default_block_size()>
        <<<default_grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
          first, num, init, op, ref);
    }
  } else {
    insert_or_apply<HasInit, CGSize, cuco::detail::default_block_size()>
      <<<default_grid_size, cuco::detail::default_block_size(), 0, stream.get()>>>(
        first, num, init, op, ref);
  }
}
}  // namespace cuco::detail::static_map_ns
