/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace cuco {
namespace detail {
namespace cg = cooperative_groups;

/**
 * @brief Initializes each slot in the flat `slot` storage.
 *
 * @tparam block_size The size of the thread block
 * @tparam atomic_slot_type Type of the slot's atomic container
 * @param slots Pointer to flat `slot` storage
 * @param num_slots Size of the storage pointed to by `slots`
 */
template <std::size_t block_size, typename atomic_slot_type>
__global__ void initialize(atomic_slot_type* const slots, std::size_t num_slots)
{
  for (std::size_t tid = block_size * blockIdx.x + threadIdx.x; tid < num_slots;
       tid += gridDim.x * block_size) {
    new (&slots[tid]) atomic_slot_type{0};
  }
}

/**
 * @brief Inserts all keys in the range `[first, last)`.
 *
 * @tparam block_size The size of the thread block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the filter's `key_type`
 * @tparam View Type of device view allowing access of filter storage
 * @tparam Hash Unary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param view Mutable device view used to access the filter's slot storage
 * @param hash The unary function to apply to hash each key
 */
template <std::size_t block_size, typename InputIt, typename View, typename Hash>
__global__ void insert(InputIt first, InputIt last, View view, Hash hash)
{
  std::size_t tid = block_size * blockIdx.x + threadIdx.x;
  auto it         = first + tid;

  while (it < last) {
    typename View::key_type const key{*it};
    view.insert(key, hash);
    it += gridDim.x * block_size;
  }
}

/**
 * @brief Indicates whether the keys in the range `[first, last)` are contained
 * in the filter.
 *
 * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists
 * in the filter.
 *
 * @tparam block_size The size of the thread block
 * @tparam InputIt Device accessible input iterator whose `value_type` is
 * convertible to the filter's `key_type`
 * @tparam OutputIt Device accessible output iterator whose `value_type` is
 * convertible to `bool`.
 * @tparam View Type of device view allowing access of filter storage
 * @tparam Hash Unary callable type
 * @param first Beginning of the sequence of keys
 * @param last End of the sequence of keys
 * @param output_begin Beginning of the sequence of booleans for the presence of each key
 * @param view Mutable device view used to access the filter's slot storage
 * @param hash The unary function to apply to hash each key
 */
template <std::size_t block_size, typename InputIt, typename OutputIt, typename View, typename Hash>
__global__ void contains(InputIt first, InputIt last, OutputIt output_begin, View view, Hash hash)
{
  std::size_t tid = block_size * blockIdx.x + threadIdx.x;
  auto it         = first + tid;
  __shared__ bool write_buffer[block_size];

  while ((first + tid) < last) {
    /*
     * The ld.relaxed.gpu instruction used in view.contains causes L1 to
     * flush more frequently, causing increased sector stores from L2 to global memory.
     * By writing results to shared memory and then synchronizing before writing back
     * to global, we no longer rely on L1, preventing the increase in sector stores from
     * L2 to global and improving performance.
     */
    typename View::key_type const key{*(first + tid)};
    write_buffer[threadIdx.x] = view.contains(key, hash);
    __syncthreads();
    *(output_begin + tid) = write_buffer[threadIdx.x];
    tid += gridDim.x * block_size;
  }
}

}  // namespace detail
}  // namespace cuco
