/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cuco/allocator.hpp>

#include <thrust/functional.h>

#include <utility>
#include <vector>

namespace cuco {

/*
 * @brief A GPU-accelerated priority queue of key-value pairs
 *
 * Allows for multiple concurrent insertions as well as multiple concurrent
 * deletions
 *
 * Current limitations:
 * - Does not support insertion and deletion at the same time
 *   - The implementation of the priority queue is based on
 *     https://arxiv.org/pdf/1906.06504.pdf, which provides a way to allow
 *     concurrent insertion and deletion, so this could be added later if useful
 * - Capacity is fixed and the queue does not automatically resize
 * - Deletion from the queue is much slower than insertion into the queue
 *   due to congestion at the underlying heap's root node
 *
 * The queue supports two operations:
 *   `push`: Add elements into the queue
 *   `pop`: Remove the element(s) with the lowest (when Max == false) or highest
 *        (when Max == true) keys
 *
 * The priority queue supports bulk host-side operations and more fine-grained
 * device-side operations.
 *
 * The host-side bulk operations `push` and `pop` allow an arbitrary number of
 * elements to be pushed to or popped from the queue.
 *
 * The device-side operations allow a cooperative group to push or pop from
 * device code. These device side
 * operations are invoked with a trivially-copyable device view,
 * `device_mutable_view` which can be obtained with the host function
 * `get_mutable_device_view` and passed to the device.
 *
 * @tparam T Type of the elements stored in the queue
 * @tparam Compare Comparison operator used to order the elements in the queue
 * @tparam Allocator Allocator defining how memory is allocated internally
 */
template <typename T,
          typename Compare   = thrust::less<T>,
          typename Allocator = cuco::cuda_allocator<char>>
class priority_queue {
  using int_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<int>;

  using t_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<T>;

  using size_t_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<size_t>;

 public:
  /**
   * @brief Construct a priority queue
   *
   * @param initial_capacity The number of elements the priority queue can hold
   * @param alloc Allocator used for allocating device storage
   */
  priority_queue(std::size_t initial_capacity,
                 Allocator const& alloc = Allocator{},
                 cudaStream_t stream    = 0);

  /**
   * @brief Push elements into the priority queue
   *
   * @tparam InputIt Device accessible input iterator whose `value_type`
   *        can be converted to T
   * @param first Beginning of the sequence of elements
   * @param last End of the sequence of elements
   * @param stream The stream in which the underlying device operations will be
   *               executed
   */
  template <typename InputIt>
  void push(InputIt first, InputIt last, cudaStream_t stream = 0);

  /**
   * @brief Remove a sequence of the lowest elements ordered by Compare
   *
   * @tparam OutputIt Device accessible output iterator whose `value_type`
   *        can be converted to T
   * @param first Beginning of the sequence of output elements
   * @param last End of the sequence of output elements
   * @param stream The stream in which the underlying GPU operations will be
   *               run
   */
  template <typename OutputIt>
  void pop(OutputIt first, OutputIt last, cudaStream_t stream = 0);

  /*
   * @brief Return the amount of shared memory required for operations
   *        on the queue with a thread block size of block_size
   *
   * @param block_size Size of the blocks to calculate storage for
   * @return The amount of temporary storage required in bytes
   */
  int get_shmem_size(int block_size)
  {
    int intersection_bytes = 2 * (block_size + 1) * sizeof(int);
    int node_bytes         = node_size_ * sizeof(T);
    return intersection_bytes + 2 * node_bytes;
  }

  /**
   * @brief Destroys the queue and frees its contents
   */
  ~priority_queue();

  class device_mutable_view {
   public:
    /**
     * @brief Push elements into the priority queue
     *
     * @tparam CG Cooperative Group type
     * @tparam InputIt Device accessible iterator whose `value_type`
     *         is convertible to T
     * @param g The cooperative group that will perform the operation
     * @param first The beginning of the sequence of elements to insert
     * @param last The end of the sequence of elements to insert
     * @param temp_storage Pointer to a contiguous section of memory
     *        large enough to hold get_shmem_size(g.size()) bytes
     */
    template <typename CG, typename InputIt>
    __device__ void push(CG const& g, InputIt first, InputIt last, void* temp_storage);

    /**
     * @brief Pop elements from the priority queue
     *
     * @tparam CG Cooperative Group type
     * @tparam OutputIt Device accessible iterator whose `value_type`
     *         is convertible to T
     * @param g The cooperative group that will perform the operation
     * @param first The beginning of the sequence of elements to output into
     * @param last The end of the sequence of elements to output into
     * @param temp_storage Pointer to a contiguous section of memory
     *        large enough to hold get_shmem_size(g.size()) bytes
     */
    template <typename CG, typename OutputIt>
    __device__ void pop(CG const& g, OutputIt first, OutputIt last, void* temp_storage);

    /*
     * @brief Return the amount of temporary storage required for operations
     * on the queue with a cooperative group size of block_size
     *
     * @param block_size Size of the cooperative groups to calculate storage for
     * @return The amount of temporary storage required in bytes
     */
    __device__ int get_shmem_size(int block_size)
    {
      int intersection_bytes = 2 * (block_size + 1) * sizeof(int);
      int node_bytes         = node_size_ * sizeof(T);
      return intersection_bytes + 2 * node_bytes;
    }

    __host__ __device__ device_mutable_view(std::size_t node_size,
                                            T* d_heap,
                                            int* d_size,
                                            std::size_t* d_p_buffer_size,
                                            int* d_locks,
                                            int lowest_level_start,
                                            int node_capacity,
                                            Compare const& compare)
      : node_size_(node_size),
        d_heap_(d_heap),
        d_size_(d_size),
        d_p_buffer_size_(d_p_buffer_size),
        d_locks_(d_locks),
        lowest_level_start_(lowest_level_start),
        node_capacity_(node_capacity),
        compare_(compare)
    {
    }

   private:
    std::size_t node_size_;   ///< Size of the heap's nodes (i.e. number of T's
                              ///  in each node)
    int lowest_level_start_;  ///< Index in `d_heap_` of the first node in the
                              ///  heap's lowest level
    int node_capacity_;       ///< Capacity of the heap in nodes

    T* d_heap_;                     ///< Pointer to an array of nodes, the 0th node
                                    ///  being the heap's partial buffer, and nodes
                                    ///  1..(node_capacity_) being the heap, where
                                    ///  the 1st node is the root
    int* d_size_;                   ///< Number of nodes currently in the heap
    std::size_t* d_p_buffer_size_;  ///< Number of elements currently in the
                                    ///  partial buffer
    int* d_locks_;                  ///< Array of locks where `d_locks_[i]` is the
                                    ///  lock for the node starting at
                                    ///  d_heap_[node_size * i]`
    Compare compare_{};             ///< Comparator used to order the elements in the queue
  };

  /*
   * @brief Returns a trivially-copyable class that can be used to perform
   *        insertion and deletion of single nodes in device code with
   *        cooperative groups
   *
   * @return A device view
   */
  device_mutable_view get_mutable_device_view()
  {
    return device_mutable_view(node_size_,
                               d_heap_,
                               d_size_,
                               d_p_buffer_size_,
                               d_locks_,
                               lowest_level_start_,
                               node_capacity_,
                               compare_);
  }

 private:
  std::size_t node_size_;   ///< Size of the heap's nodes (i.e. number of T's
                            ///  in each node)
  int lowest_level_start_;  ///< Index in `d_heap_` of the first node in the
                            ///  heap's lowest level
  int node_capacity_;       ///< Capacity of the heap in nodes

  T* d_heap_;                     ///< Pointer to an array of nodes, the 0th node
                                  ///  being the heap's partial buffer, and nodes
                                  ///  1..(node_capacity_) being the heap, where the
                                  ///  1st node is the root
  int* d_size_;                   ///< Number of nodes currently in the heap
  std::size_t* d_p_buffer_size_;  ///< Number of elements currently in the
                                  ///  partial buffer
  int* d_locks_;                  ///< Array of locks where `d_locks_[i]` is the
                                  ///  lock for the node starting at
                                  ///  d_heap_[node_size * i]`

  int_allocator_type int_allocator_;        ///< Allocator used to allocated ints
                                            ///  for example, the lock array
  t_allocator_type t_allocator_;            ///< Allocator used to allocate T's
                                            ///  and therefore nodes
  size_t_allocator_type size_t_allocator_;  ///< Allocator used to allocate
                                            ///  size_t's, e.g. d_p_buffer_size_

  Compare compare_{};  ///< Comparator used to order the elements in the queue
};

}  // namespace cuco

#include <cuco/detail/priority_queue.inl>
