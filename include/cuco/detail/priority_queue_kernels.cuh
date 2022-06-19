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

#include <cooperative_groups.h>

#include <assert.h>

namespace cuco {
namespace detail {
namespace cg = cooperative_groups;

constexpr int kPBufferIdx = 0;
constexpr int kRootIdx    = 1;

/*
 * Struct to hold pointers to the temp storage used by the priority
 * queue's kernels and functions.
 * Ideally, this temp storage is in shared memory
 */
template <typename T>
struct shared_memory_layout {
  int* intersections;
  T* a;
  T* b;
};

/*
 * Get the shared memory layout for a given group dimension
 * and node size.
 *
 * @param s Pointer to the beginning of the section of shared memory to
 *          partition
 * @param dim Size of the cooperative group the memory will be used by
 * @param node_size Size of the nodes in this priority queue
 * @returns The memory layout for the given group dimension and node size
 */
template <typename T>
__device__ shared_memory_layout<T> get_shared_memory_layout(int* s, int dim,
                                                            std::size_t node_size)
{
  shared_memory_layout<T> result;
  result.intersections = s;
  result.a             = (T*)(s + 2 * (dim + 1));
  result.b             = result.a + node_size;
  return result;
}

/**
 *  Acquires lock l for the current thread block
 *  The entire thread block must call the function
 *
 *  @param g The cooperative group that will acquire the lock
 *  @param l Pointer to the lock to be acquired
 */
template <typename CG>
__device__ void acquire_lock(CG const& g, int* l)
{
  if (g.thread_rank() == 0) {
    while (atomicCAS(l, 0, 1) != 0)
      ;
  }
  __threadfence();
  g.sync();
}

/**
 * Releases lock l for the current thread block
 *
 * @param g The cooperative group that will release the lock
 * @param l Pointer to the lock to be released
 */
template <typename CG>
__device__ void release_lock(CG const& g, int* l)
{
  if (g.thread_rank() == 0) { atomicExch(l, 0); }
}

/**
 * Copy pairs from src to dst
 *
 * @param g The cooperative group that will perform the copy
 * @param dst_start Iterator to the beginning of the destination array
 * @param src_start Iterator to the beginning of the source array
 * @param src_end Iterator to the end of the source array
 */
template <typename InputIt1, typename InputIt2, typename CG>
__device__ void copy_pairs(CG const& g, InputIt1 dst_start, InputIt2 src_start,
                           InputIt2 src_end)
{
  auto dst = dst_start + g.thread_rank();
  for (auto src = src_start + g.thread_rank(); src < src_end;
       dst += g.size(), src += g.size()) {
    *dst = *src;
  }
}

/**
 * Copy node_size pairs from src to dst
 *
 * @param g The cooperative group that will perform the copy
 * @param dst_start Iterator to the beginning of the destination array
 * @param src_start Iterator to the beginning of the source array
 * @param num_pairs Number of pairs to copy
 */
template <typename InputIt1, typename InputIt2, typename CG>
__device__ void copy_pairs(CG const& g, InputIt1 dst_start, InputIt2 src_start,
                           std::size_t num_pairs)
{
  copy_pairs(g, dst_start, src_start, src_start + num_pairs);
}

/**
 * Merge arrays a and b of size node_size by key, putting the
 * node_size elements with the lowest keys in lo, sorted by key, and the
 * node_size elements with the highest keys in hi, sorted by key
 *
 * @param g The cooperative group that will perform the merge and sort
 * @param a The first array of pairs to be merged, sorted by key
 * @param b The second array of pairs to be merged, sorted by key
 * @param lo The array in which the node_size elements with the lowest keys
 *           will be placed when the merge is completed
 * @param hi The array in which the node_size elements with the highest keys
 *           will be placed when the merge is completed
 * @param node_size The size of arrays a, b, lo, and hi
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements to be merged
 */
template <typename T, typename CG, typename Compare>
__device__ void merge_and_sort(CG const& g,
                              T* a,
                              T* b,
                              T* lo,
                              T* hi,
                              std::size_t node_size,
                              shared_memory_layout<T> shmem,
                              Compare const& compare)
{
  merge_and_sort(g, a, b, lo, hi, node_size, node_size, node_size, shmem, compare);
}

/**
 * Merge array a of size num_elements_a and array b of size num_elements_b
 * by key. If num_elements_a + num_elements_b <= node_size, all merged elements
 * will be placed in lo. Otherwise, the node_size lowest merged elements will
 * be placed in lo, and the rest of the elements will be placed in hi.
 *
 * @param g The cooperative group that will perform the merge and sort
 * @param a The first array of pairs to be merged, sorted by key
 * @param b The second array of pairs to be merged, sorted by key
 * @param lo The array in which the node_size elements with the lowest keys
 *           will be placed when the merge is completed
 * @param hi The array in which the node_size elements with the highest keys
 *           will be placed when the merge is completed,
 *           if num_elements_a + num_elements_b > node_size. May be nullptr in
 *           the case that num_elements_a + num_elements_b < node_size.
 * @param num_elements_a The number of pairs in array a
 * @param num_elements_b The number of pairs in array b
 * @param node_size The size of arrays hi and lo, in other words how many
 *                  elements to insert into lo before starting insertion into
 *                  hi
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements to be merged
 */
template <typename T, typename CG, typename Compare>
__device__ void merge_and_sort(CG const& g,
                               T* a,
                               T* b,
                               T* lo,
                               T* hi,
                               std::size_t num_elements_a,
                               std::size_t num_elements_b,
                               std::size_t node_size,
                               shared_memory_layout<T> shmem,
                               Compare const& compare)
{
  const int lane = g.thread_rank();
  const int dim  = g.size();

  if (num_elements_a == node_size && compare(a[node_size - 1], b[0])) {
    copy_pairs(g, lo, a, num_elements_a);

    copy_pairs(g, hi, b, num_elements_b);
    return;
  }

  if (num_elements_b == node_size && compare(b[node_size - 1], a[0])) {
    copy_pairs(g, hi, a, num_elements_a);

    copy_pairs(g, lo, b, num_elements_b);
    return;
  }

  // Array of size 2 * (blockDim.x + 1)
  int* const intersections = shmem.intersections;

  if (lane == 0) {
    intersections[0] = 0;
    intersections[1] = 0;

    intersections[2 * dim]     = node_size;
    intersections[2 * dim + 1] = node_size;
  }

  // Calculate the diagonal spacing
  const int p = 2 * node_size / dim;

  // There will be one less diagonal than threads
  if (threadIdx.x != 0) {
    // i + j = (p * threadIdx.x - 1)
    const int j_bl = min((int)node_size - 1, p * lane - 1);
    const int i_bl = (p * lane - 1) - j_bl;

    const int diag_len = min(p * lane, (int)node_size - i_bl);

    // Will be the location of the rightmost one
    // in the merge-path grid in terms of array a
    int rightmost_one = i_bl - 1;

    // Location of leftmost zero
    int leftmost_zero = i_bl + diag_len;

    // Binary search along the diagonal
    while (leftmost_zero - rightmost_one > 1) {
      const int i = (rightmost_one + leftmost_zero) / 2;
      const int j = (p * lane - 1) - i;

      if (i >= num_elements_a) {
        leftmost_zero = i;
      } else if (j >= num_elements_b || compare(a[i], b[j])) {
        rightmost_one = i;
      } else {
        leftmost_zero = i;
      }
    }

    intersections[2 * lane]     = leftmost_zero;
    intersections[2 * lane + 1] = (p * lane - 1) - leftmost_zero + 1;
  }

  g.sync();

  // Get the intersection that starts this partition
  int i = intersections[2 * lane];
  int j = intersections[2 * lane + 1];

  // Get the intersection that ends this partition
  const int i_max = min(intersections[2 * (lane + 1)], (int)num_elements_a);
  const int j_max = min(intersections[2 * (lane + 1) + 1], (int)num_elements_b);

  // Insert location into the output array
  int ins_loc = lane * p;

  // Merge our partition into the output arrays
  while (i < i_max && j < j_max) {
    T next_element;
    if (compare(a[i], b[j])) {
      next_element = a[i];
      i++;
    } else {
      next_element = b[j];
      j++;
    }
    if (ins_loc < node_size) {
      lo[ins_loc] = next_element;
    } else {
      hi[ins_loc - node_size] = next_element;
    }
    ins_loc++;
  }

  // Insert the any remaining elements in a
  while (i < i_max) {
    if (ins_loc < node_size) {
      lo[ins_loc] = a[i];
      i++;
    } else {
      hi[ins_loc - node_size] = a[i];
      i++;
    }
    ins_loc++;
  }

  // Insert any remaining elements in b
  while (j < j_max) {
    if (ins_loc < node_size) {
      lo[ins_loc] = b[j];
      j++;
    } else {
      hi[ins_loc - node_size] = b[j];
      j++;
    }
    ins_loc++;
  }
}

/**
 * Sorts the len pairs at start by key
 *
 * @param g The cooperative group that will perform the sort
 * @param start Pointer to the array to be sorted
 * @param len Number of pairs to be sorted
 * @param node_size A power of two corresponding to the number of pairs
 *                  temp can contain
 * @param temp A temporary array containing space for at least the nearest
 *             power of two greater than len pairs
 * @param compare Comparison operator ordering the elements to be sorted
 */
template <typename T, typename CG, typename Compare>
__device__ void pb_sort(
  CG const& g, T* start, std::size_t len, std::size_t node_size, T* temp,
  Compare const& compare)
{
  const int lane = g.thread_rank();
  const int dim  = g.size();

  char* const mask = (char*)temp;

  for (int i = lane; i < node_size; i += dim) {
    mask[i] = i < len;
  }
  g.sync();

  // Build a bitonic sequence
  for (int width = 2; width < node_size; width *= 2) {
    for (int jump = width / 2; jump >= 1; jump /= 2) {
      for (int i = lane; i < node_size / 2; i += dim) {
        const int start_jump = width / 2;
        const int left       = (i / jump) * jump * 2 + i % jump;
        const int right      = left + jump;
        if ((i / start_jump) % 2 == 0) {
          if (!mask[left] || 
              (mask[right] && !compare(start[left], start[right]))) {
            auto temp    = start[left];
            start[left]  = start[right];
            start[right] = temp;

            auto temp_mask = mask[left];
            mask[left]     = mask[right];
            mask[right]    = temp_mask;
          }
        } else {
          if (!mask[right] ||
              (mask[left] && compare(start[left], start[right]))) {
            auto temp    = start[left];
            start[left]  = start[right];
            start[right] = temp;

            auto temp_mask = mask[left];
            mask[left]     = mask[right];
            mask[right]    = temp_mask;
          }
        }
      }
      g.sync();
    }
  }

  // Merge to get the sorted result
  for (int jump = node_size / 2; jump >= 1; jump /= 2) {
    for (int i = lane; i < node_size / 2; i += dim) {
      const int left  = (i / jump) * jump * 2 + i % jump;
      const int right = left + jump;
      if (!mask[left] || (mask[right] && !compare(start[left], start[right]))) {
        auto temp    = start[left];
        start[left]  = start[right];
        start[right] = temp;

        auto temp_mask = mask[left];
        mask[left]     = mask[right];
        mask[right]    = temp_mask;
      }
    }
    g.sync();
  }
}

/**
 * Reverses the bits after the most significant set bit in x
 * i.e. if x is 1abc..xyz in binary returns 1zyx...cba
 *
 * @param x The number whose lower bits will be reversed
 * @return The number with all bits after the most significant
 *         set bit reversed
 */
__device__ int bit_reverse_perm(int x)
{
  const int clz = __clz(x);

  const int bits     = sizeof(int) * 8;
  const int high_bit = 1 << ((bits - 1) - clz);
  const int mask     = high_bit - 1;

  const int masked = x & mask;
  const int rev    = __brev(masked) >> (clz + 1);

  return high_bit | rev;
}

/**
 * Given x, the idx of a node, return when that node is inserted,
 * i.e. if x is 6 and lowest_level_start > 6, return 5 since the node
 * at element 6 will be the 5th to be inserted with the bit reversal
 * permutation. This operation is its own inverse.
 *
 * @param x The index to operate on
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 */
__device__ int insertion_order_index(int x, int lowest_level_start)
{
  assert(x > 0);

  if (x >= lowest_level_start) { return x; }

  return bit_reverse_perm(x);
}

/**
 * Find the index of the parent of the node at index x
 *
 * @param x The index to operate on
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @return The index of the parent of x
 */
__device__ int parent(int x, int lowest_level_start)
{
  assert(x > 0);
  if (x >= lowest_level_start) { return bit_reverse_perm(x) / 2; }

  return x / 2;
}

/**
 * Find the index of the left child of the node at index x
 *
 * @param x The index to operate on
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @return The index of the left child of x
 */
__device__ int left_child(int x, int lowest_level_start)
{
  assert(x > 0);
  int result = x * 2;

  if (result >= lowest_level_start) { result = bit_reverse_perm(result); }

  return result;
}

/**
 * Find the index of the right child of the node at index x
 *
 * @param x The index to operate on
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @return The index of the right child of x
 */
__device__ int right_child(int x, int lowest_level_start)
{
  assert(x > 0);
  int result = x * 2 + 1;

  if (result >= lowest_level_start) { result = bit_reverse_perm(result); }

  return result;
}

/**
 * swim node cur_node up the heap
 * Pre: g must hold the lock corresponding to cur_node
 *
 * @param g The cooperative group that will perform the operation
 * @param cur_node Index of the node to swim
 * @param heap The array of pairs that stores the heap itself
 * @param size Pointer to the number of pairs currently in the heap
 * @param node_size Size of the nodes in the heap
 * @param locks Array of locks, one for each node in the heap
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements in the heap
 */
template <typename T, typename CG, typename Compare>
__device__ void swim(CG const& g,
                     int cur_node,
                     T* heap,
                     int* size,
                     std::size_t node_size,
                     int* locks,
                     int lowest_level_start,
                     shared_memory_layout<T> shmem,
                     Compare const& compare)
{
  const int lane = g.thread_rank();
  const int dim  = g.size();

  int cur_parent = parent(cur_node, lowest_level_start);

  // swim the new node up the tree
  while (cur_node != 1) {
    acquire_lock(g, &(locks[cur_parent]));

    // If the heap property is already satisfied for this node and its
    // parent we are done
    if (!compare(heap[cur_node * node_size],
                 heap[cur_parent * node_size + node_size - 1])) {
      release_lock(g, &(locks[cur_parent]));
      break;
    }

    merge_and_sort(g,
                   &heap[cur_parent * node_size],
                   &heap[cur_node * node_size],
                   shmem.a,
                   shmem.b,
                   node_size,
                   shmem,
                   compare);

    g.sync();

    copy_pairs(g, &heap[cur_parent * node_size], shmem.a, node_size);
    copy_pairs(g, &heap[cur_node * node_size], shmem.b, node_size);

    g.sync();

    release_lock(g, &(locks[cur_node]));
    cur_node = cur_parent;
    cur_parent   = parent(cur_node, lowest_level_start);
  }

  release_lock(g, &(locks[cur_node]));
}

/**
 * sink the root down the heap
 * Pre: g must hold the root's lock
 *
 * @param g The cooperative group that will perform the operation
 * @param heap The array of pairs that stores the heap itself
 * @param size Pointer to the number of pairs currently in the heap
 * @param node_size Size of the nodes in the heap
 * @param locks Array of locks, one for each node in the heap
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @param node_capacity Max capacity of the heap in nodes
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements in the heap
 */
template <typename T, typename CG, typename Compare>
__device__ void sink(CG const& g,
                     T* heap,
                     int* size,
                     std::size_t node_size,
                     int* locks,
                     std::size_t* p_buffer_size,
                     int lowest_level_start,
                     int node_capacity,
                     shared_memory_layout<T> shmem,
                     Compare const& compare)
{
  std::size_t cur = kRootIdx;

  const int dim = g.size();

  // sink the node
  while (insertion_order_index(left_child(cur, lowest_level_start),
                               lowest_level_start) <= node_capacity) {
    const std::size_t left  = left_child(cur, lowest_level_start);
    const std::size_t right = right_child(cur, lowest_level_start);

    acquire_lock(g, &locks[left]);

    // The left node might have been removed
    // since the while loop condition, in which
    // case we are already at the bottom of the heap
    if (insertion_order_index(left, lowest_level_start) > *size) {
      release_lock(g, &locks[left]);
      break;
    }

    std::size_t lo;

    if (insertion_order_index(right, lowest_level_start) <= node_capacity) {
      acquire_lock(g, &locks[right]);

      // Note that even with the bit reversal permutation,
      // we can never have a right child without a left child
      //
      // If we have both children, merge and sort them
      if (insertion_order_index(right, lowest_level_start) <= *size) {
        std::size_t hi;

        // In order to ensure we preserve the heap property,
        // we put the largest node_size elements in the child
        // that previously contained the largest element
        if (!compare(heap[(left + 1) * node_size - 1],
                    heap[(right + 1) * node_size - 1])) {
          hi = left;
          lo = right;
        } else {
          lo = left;
          hi = right;
        }

        // Skip the merge and sort if the nodes are already correctly
        // sorted
        if (!compare(heap[(lo + 1) * node_size - 1], heap[hi * node_size])) {
          merge_and_sort(g,
                         &heap[left * node_size],
                         &heap[right * node_size],
                         shmem.a,
                         shmem.b,
                         node_size,
                         shmem,
                         compare);

          g.sync();

          copy_pairs(g, &heap[hi * node_size], shmem.b, node_size);
          copy_pairs(g, &heap[lo * node_size], shmem.a, node_size);

          g.sync();
        }
        release_lock(g, &locks[hi]);
      } else {
        lo = left;
        release_lock(g, &locks[right]);
      }
    } else {
      lo = left;
    }

    merge_and_sort(g,
                   &heap[lo * node_size],
                   &heap[cur * node_size],
                   shmem.a,
                   shmem.b,
                   node_size,
                   shmem,
                   compare);

    g.sync();

    copy_pairs(g, &heap[lo * node_size], shmem.b, node_size);
    copy_pairs(g, &heap[cur * node_size], shmem.a, node_size);

    g.sync();

    release_lock(g, &locks[cur]);

    cur = lo;
  }
  release_lock(g, &locks[cur]);
}

/**
 * Add exactly node_size elements into the heap from
 * elements
 *
 * @param g The cooperative group that will perform the push
 * @param elements Iterator for the elements to be inserted
 * @param heap The array of pairs that stores the heap itself
 * @param size Pointer to the number of pairs currently in the heap
 * @param node_size Size of the nodes in the heap
 * @param locks Array of locks, one for each node in the heap
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements in the heap
 */
template <typename InputIt, typename T, typename Compare, typename CG>
__device__ void push_single_node(CG const& g,
                                 InputIt elements,
                                 T* heap,
                                 int* size,
                                 std::size_t node_size,
                                 int* locks,
                                 int lowest_level_start,
                                 shared_memory_layout<T> shmem,
                                 Compare const& compare)
{
  const int lane = g.thread_rank();
  const int dim  = g.size();

  copy_pairs(g, shmem.a, elements, elements + node_size);

  g.sync();

  pb_sort(g, shmem.a, node_size, node_size, shmem.b, compare);

  int* const cur_node_temp = (int*)shmem.intersections;
  if (lane == 0) { *cur_node_temp = atomicAdd(size, 1) + 1; }
  g.sync();

  const int cur_node = insertion_order_index(*cur_node_temp, lowest_level_start);

  acquire_lock(g, &(locks[cur_node]));

  copy_pairs(g, &heap[cur_node * node_size], shmem.a, node_size);

  g.sync();

  swim(g, cur_node, heap, size, node_size, locks, lowest_level_start,
       shmem, compare);
}

/**
 * Remove exactly node_size elements from the heap and place them
 * in elements
 *
 * @param g The cooperative group that will perform the pop
 * @param elements Iterator to the elements to write to
 * @param heap The array of pairs that stores the heap itself
 * @param size Pointer to the number of pairs currently in the heap
 * @param node_size Size of the nodes in the heap
 * @param locks Array of locks, one for each node in the heap
 * @param p_buffer_size Number of pairs in the heap's partial buffer
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @param node_capacity Maximum capacity of the heap in nodes
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements in the heap
 */
template <typename OutputIt, typename T, typename Compare, typename CG>
__device__ void pop_single_node(CG const& g,
                                OutputIt elements,
                                T* heap,
                                int* size,
                                std::size_t node_size,
                                int* locks,
                                std::size_t* p_buffer_size,
                                int lowest_level_start,
                                int node_capacity,
                                shared_memory_layout<T> shmem,
                                Compare const& compare)
{
  const int lane = g.thread_rank();
  const int dim  = g.size();

  acquire_lock(g, &locks[kRootIdx]);
  if (*size == 0) {
    copy_pairs(g, elements, heap, node_size);

    if (lane == 0) { *p_buffer_size = 0; }
    g.sync();
    return;
  }

  // Find the target node (the last one inserted) and
  // decrement the size

  const std::size_t tar = insertion_order_index(*size, lowest_level_start);

  if (tar != 1) { acquire_lock(g, &locks[tar]); }

  g.sync();

  if (lane == 0) { *size -= 1; }
  g.sync();

  // Copy the root to the output array

  copy_pairs(g, elements, &heap[node_size], &heap[node_size] + node_size);

  g.sync();

  // Copy the target node to the root

  if (tar != kRootIdx) {
    copy_pairs(g, &heap[node_size], &heap[tar * node_size], node_size);

    release_lock(g, &locks[tar]);

    g.sync();
  }

  // Merge and sort the root and the partial buffer

  merge_and_sort(g,
                 &heap[node_size],
                 &heap[kPBufferIdx],
                 shmem.a,
                 shmem.b,
                 node_size,
                 *p_buffer_size,
                 node_size,
                 shmem,
                 compare);

  g.sync();

  copy_pairs(g, &heap[node_size], shmem.a, node_size);

  copy_pairs(g, heap, shmem.b, *p_buffer_size);

  g.sync();

  sink(g,
       heap,
       size,
       node_size,
       locks,
       p_buffer_size,
       lowest_level_start,
       node_capacity,
       shmem,
       compare);
}

/**
 * Remove num_elements < node_size elements from the heap and place them
 * in elements
 *
 * @param elements The array of elements to insert into
 * @param num_elements The number of elements to remove
 * @param heap The array of pairs that stores the heap itself
 * @param size Pointer to the number of pairs currently in the heap
 * @param node_size Size of the nodes in the heap
 * @param locks Array of locks, one for each node in the heap
 * @param p_buffer_size Number of pairs in the heap's partial buffer
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @param node_capacity Maximum capacity of the heap in nodes
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements in the heap
 */
template <typename InputIt, typename T, typename Compare, typename CG>
__device__ void pop_partial_node(CG const& g,
                                 InputIt elements,
                                 std::size_t num_elements,
                                 T* heap,
                                 int* size,
                                 std::size_t node_size,
                                 int* locks,
                                 std::size_t* p_buffer_size,
                                 int lowest_level_start,
                                 int node_capacity,
                                 shared_memory_layout<T> shmem,
                                 Compare const& compare)
{
  const int lane = g.thread_rank();
  const int dim  = g.size();

  acquire_lock(g, &locks[kRootIdx]);

  if (*size == 0) {
    copy_pairs(g, elements, heap, num_elements);
    g.sync();

    const std::size_t n_p_buffer_size = *p_buffer_size - num_elements;

    copy_pairs(g, shmem.a, heap + num_elements, n_p_buffer_size);

    g.sync();

    copy_pairs(g, heap, shmem.a, n_p_buffer_size);

    if (lane == 0) { *p_buffer_size = n_p_buffer_size; }

    release_lock(g, &locks[kRootIdx]);
  } else {
    copy_pairs(g, elements, &heap[kRootIdx * node_size], num_elements);
    g.sync();

    if (*p_buffer_size >= num_elements) {
      merge_and_sort(g,
                   &heap[kPBufferIdx],
                   &heap[kRootIdx * node_size] + num_elements,
                   shmem.a,
                   shmem.b,
                   *p_buffer_size,
                   node_size - num_elements,
                   node_size,
                   shmem,
                   compare);

      g.sync();

      if (lane == 0) { *p_buffer_size = *p_buffer_size - num_elements; }

      g.sync();

      copy_pairs(g, &heap[kRootIdx * node_size], shmem.a, node_size);
      copy_pairs(g, &heap[kPBufferIdx], shmem.b, *p_buffer_size);

      g.sync();

      sink(g,
           heap,
           size,
           node_size,
           locks,
           p_buffer_size,
           lowest_level_start,
           node_capacity,
           shmem,
           compare);
    } else {
      merge_and_sort(g,
                   &heap[kPBufferIdx],
                   &heap[kRootIdx * node_size] + num_elements,
                   shmem.a,
                   (T*)nullptr,
                   *p_buffer_size,
                   node_size - num_elements,
                   node_size,
                   shmem,
                   compare);

      g.sync();

      copy_pairs(g, &heap[kPBufferIdx], shmem.a,
           	     *p_buffer_size + node_size - num_elements);

      const int tar = insertion_order_index(*size, lowest_level_start);
      g.sync();

      *p_buffer_size += node_size;
      *p_buffer_size -= num_elements;

      g.sync();

      if (lane == 0) { *size -= 1; }

      if (tar != kRootIdx) {
        acquire_lock(g, &locks[tar]);

        copy_pairs(g, &heap[kRootIdx * node_size], &heap[tar * node_size],
            	   node_size);

        g.sync();

        release_lock(g, &locks[tar]);

        merge_and_sort(g,
                       &heap[node_size],
                       &heap[kPBufferIdx],
                       shmem.a,
                       shmem.b,
                       node_size,
                       *p_buffer_size,
                       node_size,
                       shmem,
                       compare);
        g.sync();

        copy_pairs(g, &heap[node_size], shmem.a, node_size);

        copy_pairs(g, heap, shmem.b, *p_buffer_size);

        g.sync();

        sink(g,
             heap,
             size,
             node_size,
             locks,
             p_buffer_size,
             lowest_level_start,
             node_capacity,
             shmem,
             compare);
      } else {
        release_lock(g, &locks[kRootIdx]);
      }
    }
  }
}

/**
 * Add p_ins_size < node_size elements into the heap from
 * elements
 *
 * @param g The cooperative group that will perform the push
 * @param elements The array of elements to add
 * @param p_ins_size The number of elements to be inserted
 * @param heap The array of pairs that stores the heap itself
 * @param size Pointer to the number of pairs currently in the heap
 * @param node_size Size of the nodes in the heap
 * @param locks Array of locks, one for each node in the heap
 * @param p_buffer_size The size of the partial buffer
 * @param lowest_level_start Index of the first node in the last level of the
 *                           heap
 * @param shmem The shared memory layout for this cooperative group
 * @param compare Comparison operator ordering the elements in the heap
 */
template <typename InputIt, typename T, typename Compare, typename CG>
__device__ void push_partial_node(CG const& g,
                                  InputIt elements,
                                  std::size_t p_ins_size,
                                  T* heap,
                                  int* size,
                                  std::size_t node_size,
                                  int* locks,
                                  std::size_t* p_buffer_size,
                                  int lowest_level_start,
                                  shared_memory_layout<T> shmem,
                                  Compare const& compare)
{
  const int lane = g.thread_rank();
  const int dim  = g.size();

  acquire_lock(g, &locks[kRootIdx]);

  copy_pairs(g, shmem.b, elements, p_ins_size);

  pb_sort(g, shmem.b, p_ins_size, node_size, shmem.a, compare);

  // There is enough data for a new node, in which case we
  // construct a new node and insert it
  if (*p_buffer_size + p_ins_size >= node_size) {
    int* const cur_node_temp = shmem.intersections;
    if (lane == 0) { *cur_node_temp = atomicAdd(size, 1) + 1; }
    g.sync();

    const int cur_node = insertion_order_index(*cur_node_temp,
             	                               lowest_level_start);

    if (cur_node != kRootIdx) { acquire_lock(g, &(locks[cur_node])); }

    g.sync();

    merge_and_sort(g,
                   shmem.b,
                   &heap[kPBufferIdx],
                   &heap[cur_node * node_size],
                   shmem.a,
                   p_ins_size,
                   *p_buffer_size,
                   node_size,
                   shmem,
                   compare);

    if (lane == 0) { *p_buffer_size = (*p_buffer_size + p_ins_size) - node_size; }

    g.sync();

    copy_pairs(g, heap, shmem.a, *p_buffer_size);

    if (cur_node != kRootIdx) { release_lock(g, &locks[kRootIdx]); }

    swim(g, cur_node, heap, size, node_size, locks, lowest_level_start, shmem, compare);

  } else {
    // There are not enough elements for a new node,
    // in which case we merge and sort the root and
    // the elements to be inserted and then the root
    // and the partial buffer

    merge_and_sort(g,
                   shmem.b,
                   &heap[kPBufferIdx],
                   shmem.a,
                   (T*)nullptr,
                   p_ins_size,
                   *p_buffer_size,
                   node_size,
                   shmem,
                   compare);

    g.sync();

    if (lane == 0) { *p_buffer_size += p_ins_size; }

    g.sync();

    copy_pairs(g, heap, shmem.a, *p_buffer_size);

    g.sync();

    if (*size > 0) {
      merge_and_sort(g,
                     &heap[node_size],
                     &heap[kPBufferIdx],
                     shmem.a,
                     shmem.b,
                     node_size,
                     *p_buffer_size,
                     node_size,
                     shmem,
                     compare);
      g.sync();

      copy_pairs(g, heap, shmem.b, *p_buffer_size);

      copy_pairs(g, &heap[node_size], shmem.a, node_size);

      g.sync();
    }
    release_lock(g, &locks[kRootIdx]);
  }
}

/**
* Add num_elements elements into the heap from
* elements
* @param elements The array of elements to add
* @param num_elements The number of elements to be inserted
* @param heap The array of pairs that stores the heap itself
* @param size Pointer to the number of pairs currently in the heap
* @param node_size Size of the nodes in the heap
* @param locks Array of locks, one for each node in the heap
* @param p_buffer_size Number of pairs in the heap's partial buffer
* @param temp_node A temporary array large enough to store
                   sizeof(T) * node_size bytes
* @param lowest_level_start The first index of the heaps lowest layer
* @param compare Comparison operator ordering the elements in the heap
*/
template <typename OutputIt, typename T, typename Compare>
__global__ void push_kernel(OutputIt elements,
                            std::size_t num_elements,
                            T* heap,
                            int* size,
                            std::size_t node_size,
                            int* locks,
                            std::size_t* p_buffer_size,
                            int lowest_level_start,
                            Compare compare)
{
  extern __shared__ int s[];

  const shared_memory_layout<T> shmem = get_shared_memory_layout<T>(s,
                                                                    blockDim.x,
                                                                    node_size);

  // We push as many elements as possible as full nodes,
  // then deal with the remaining elements as a partial insertion
  // below
  cg::thread_block g = cg::this_thread_block();
  for (std::size_t i = blockIdx.x * node_size; i + node_size <= num_elements;
       i += gridDim.x * node_size) {
    push_single_node(
      g, elements + i, heap, size, node_size, locks, lowest_level_start, shmem, compare);
  }

  // We only need one block for partial insertion
  if (blockIdx.x != 0) { return; }

  // If node_size does not divide num_elements, there are some leftover
  // elements for which we must perform a partial insertion
  const std::size_t first_not_inserted = (num_elements / node_size) * node_size;

  if (first_not_inserted < num_elements) {
    const std::size_t p_ins_size = num_elements - first_not_inserted;
    push_partial_node(g,
                    elements + first_not_inserted,
                    p_ins_size,
                    heap,
                    size,
                    node_size,
                    locks,
                    p_buffer_size,
                    lowest_level_start,
                    shmem,
                    compare);
  }
}

/**
 * Remove exactly node_size elements from the heap and place them
 * in elements
 * @param elements The array of elements to insert into
 * @param num_elements The number of elements to remove
 * @param heap The array of pairs that stores the heap itself
 * @param size Pointer to the number of pairs currently in the heap
 * @param node_size Size of the nodes in the heap
 * @param locks Array of locks, one for each node in the heap
 * @param p_buffer_size Number of pairs in the heap's partial buffer
 * @param lowest_level_start The first index of the heaps lowest layer
 * @param node_capacity The capacity of the heap in nodes
 * @param compare Comparison operator ordering the elements in the heap
 */
template <typename OutputIt, typename T, typename Compare>
__global__ void pop_kernel(OutputIt elements,
                           std::size_t num_elements,
                           T* heap,
                           int* size,
                           std::size_t node_size,
                           int* locks,
                           std::size_t* p_buffer_size,
                           int lowest_level_start,
                           int node_capacity,
                           Compare compare)
{
  extern __shared__ int s[];

  const shared_memory_layout<T> shmem = get_shared_memory_layout<T>(s,
                                                                    blockDim.x,
                                                                    node_size);

  cg::thread_block g = cg::this_thread_block();
  for (std::size_t i = blockIdx.x; i < num_elements / node_size;
       i += gridDim.x) {
    pop_single_node(g,
                    elements + i * node_size,
                    heap,
                    size,
                    node_size,
                    locks,
                    p_buffer_size,
                    lowest_level_start,
                    node_capacity,
                    shmem,
                    compare);
  }

  // We only need one block for partial deletion
  if (blockIdx.x != 0) { return; }

  // If node_size does not divide num_elements, there are some leftover
  // elements for which we must perform a partial deletion
  const std::size_t first_not_inserted = (num_elements / node_size) * node_size;

  if (first_not_inserted < num_elements) {
    const std::size_t p_del_size = num_elements - first_not_inserted;
    pop_partial_node(g,
                     elements + first_not_inserted,
                     p_del_size,
                     heap,
                     size,
                     node_size,
                     locks,
                     p_buffer_size,
                     lowest_level_start,
                     node_capacity,
                     shmem,
                     compare);
  }
}

}  // namespace detail

}  // namespace cuco
