#pragma once

#include <cooperative_groups.h>
#include <assert.h>

using namespace cooperative_groups;

namespace cuco {

constexpr int kPBufferIdx = 0;
constexpr int kRootIdx = 1;

/*
* Struct to hold pointers to the temp storage used by the priority
* queue's kernels and functions.
* Ideally, this temp storage is in shared memory
*/
template <typename T>
struct SharedMemoryLayout {
  int *intersections;
  T *A;
  T *B;
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
__device__ SharedMemoryLayout<T> GetSharedMemoryLayout(
           int *s, int dim, size_t node_size) {

  SharedMemoryLayout<T> result;
  result.intersections = s;
  result.A = (T*)(s + 2 * (dim + 1));
  result.B = result.A + node_size;
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
__device__ void AcquireLock(CG const& g, int *l) {
  if (g.thread_rank() == 0) {
    while (atomicCAS(l, 0, 1) != 0);
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
__device__ void ReleaseLock(CG const& g, int *l) {
  if (g.thread_rank() == 0) {
    atomicExch(l, 0);
  }
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
__device__ void CopyPairs(CG const& g, InputIt1 dst_start,
                          InputIt2 src_start, InputIt2 src_end) {
  auto dst = dst_start + g.thread_rank();
  for (auto src = src_start + g.thread_rank();
       src < src_end; dst += g.size(), src += g.size()) {
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
__device__ void CopyPairs(CG const& g, InputIt1 dst_start,
                          InputIt2 src_start, size_t num_pairs) {
  CopyPairs(g, dst_start, src_start, src_start + num_pairs);
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
*/
template <typename T, typename CG, typename Compare>
__device__ void MergeAndSort(CG const& g,
                               T *a,
                               T *b,
                               T *lo,
                               T *hi,
                               size_t node_size,
                               SharedMemoryLayout<T> shmem,
			       Compare const& compare) {
  MergeAndSort(g, a, b, lo, hi, node_size,
	       node_size, node_size, shmem, compare);
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
*/
template <typename T, typename CG, typename Compare>
__device__ void MergeAndSort(CG const& g,
                               T *a,
                               T *b,
                               T *lo,
                               T *hi,
                               size_t num_elements_a,
                               size_t num_elements_b,
                               size_t node_size,
                               SharedMemoryLayout<T> shmem,
			       Compare const& compare) {

  int lane = g.thread_rank();
  int dim = g.size();

  if (num_elements_a == node_size &&
      compare(a[node_size - 1], b[0])) {

    CopyPairs(g, lo, a, num_elements_a);

    CopyPairs(g, hi, b, num_elements_b);
    return;
  }

  if (num_elements_b == node_size &&
      compare(b[node_size - 1], a[0])) {

    CopyPairs(g, hi, a, num_elements_a);

    CopyPairs(g, lo, b, num_elements_b);
    return;
  }

  // Array of size 2 * (blockDim.x + 1)
  int *intersections = shmem.intersections;


  if (lane == 0) {
    intersections[0] = 0;
    intersections[1] = 0;

    intersections[2 * dim] = node_size;
    intersections[2 * dim + 1] = node_size;
  }

  // Calculate the diagonal spacing
  int p = 2 * node_size / dim;

  // There will be one less diagonal than threads
  if (threadIdx.x != 0) {
    // i + j = (p * threadIdx.x - 1)
    int j_bl = min((int)node_size - 1, p * lane - 1);
    int i_bl = (p * lane - 1) - j_bl;

    int diag_len = min(p * lane, (int)node_size - i_bl);

    // Will be the location of the rightmost one
    // in the merge-path grid in terms of array a
    int rightmost_one = i_bl - 1;

    // Location of leftmost zero
    int leftmost_zero = i_bl + diag_len;

    // Binary search along the diagonal
    while (leftmost_zero - rightmost_one > 1) {

      int i = (rightmost_one + leftmost_zero) / 2;
      int j = (p * lane - 1) - i;

      if (i >= num_elements_a) {
        leftmost_zero = i;
      } else if (j >= num_elements_b || compare(a[i], b[j])) {
        rightmost_one = i;
      } else {
        leftmost_zero = i;
      }

    }

    intersections[2 * lane] = leftmost_zero;
    intersections[2 * lane + 1] = (p * lane - 1)
                                         - leftmost_zero + 1;

  }

  g.sync();

  // Get the intersection that starts this partition
  int i = intersections[2 * lane];
  int j = intersections[2 * lane + 1];

  // Get the intersection that ends this partition
  int i_max = min(intersections[2 * (lane + 1)], (int)num_elements_a);
  int j_max = min(intersections[2 * (lane + 1) + 1],
                  (int)num_elements_b);

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
*/
template <typename T, typename CG, typename Compare>
__device__ void PBSort(CG const& g, T *start, size_t len,
                              size_t node_size,
                              T *temp,
			      Compare const& compare) {


  int lane = g.thread_rank();
  int dim = g.size();

  char *mask = (char*)temp;

  for (int i = lane; i < node_size; i += dim) {
    mask[i] = i < len;
  }
  g.sync();

  // Build a bitonic sequence
  for (int width = 2; width < node_size; width *= 2) {
    for (int jump = width / 2; jump >= 1; jump /= 2) {
      for (int i = lane; i < node_size / 2; i += dim) {
        int start_jump = width / 2;
        int left = (i / jump) * jump * 2 + i % jump;
        int right = left + jump;
        if ((i / start_jump) % 2 == 0) {
          if (!mask[left] || (mask[right] &&
              !compare(start[left], start[right]))) {
            auto temp = start[left];
            start[left] = start[right];
            start[right] = temp;

            auto temp_mask = mask[left];
            mask[left] = mask[right];
            mask[right] = temp_mask;
          }
        } else {
          if (!mask[right] || (mask[left]
              && compare(start[left], start[right]))) {
            auto temp = start[left];
            start[left] = start[right];
            start[right] = temp;

            auto temp_mask = mask[left];
            mask[left] = mask[right];
            mask[right] = temp_mask;
          }
        }
      }
      g.sync();
    }
  }

  // Merge to get the sorted result
  for (int jump = node_size / 2; jump >= 1; jump /= 2) {
    for (int i = lane; i < node_size / 2; i += dim) {
      int left = (i / jump) * jump * 2 + i % jump;
      int right = left + jump;
      if (!mask[left] || (mask[right]
          && !compare(start[left], start[right]))) {
        auto temp = start[left];
        start[left] = start[right];
        start[right] = temp;

        auto temp_mask = mask[left];
        mask[left] = mask[right];
        mask[right] = temp_mask;
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
__device__ int BitReversePerm(int x) {
  int clz = __clz(x);

  int bits = sizeof(int) * 8;
  int high_bit = 1 << ((bits - 1) - clz);
  int mask = high_bit - 1;

  int masked = x & mask;
  int rev = __brev(masked) >> (clz + 1);

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
__device__ int InsertionOrderIndex(int x, int lowest_level_start) {
  assert(x > 0);

  if (x >= lowest_level_start) {
    return x;
  }

  return BitReversePerm(x);
}

/**
* Find the index of the parent of the node at index x
*
* @param x The index to operate on
* @param lowest_level_start Index of the first node in the last level of the
*                           heap
* @return The index of the parent of x
*/
__device__ int Parent(int x, int lowest_level_start) {

  assert(x > 0);
  if (x >= lowest_level_start) {
    return BitReversePerm(x) / 2;
  }

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
__device__ int LeftChild(int x, int lowest_level_start) {
  assert(x > 0);
  int result = x * 2;

  if (result >= lowest_level_start) {
    result = BitReversePerm(result);
  }

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
__device__ int RightChild(int x, int lowest_level_start) {

  assert(x > 0);
  int result = x * 2 + 1;

  if (result >= lowest_level_start) {
    result = BitReversePerm(result);
  }

  return result;
}

/**
* Swim node cur_node up the heap
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
*/
template <typename T, typename CG, typename Compare>
__device__ void Swim(CG const& g,
                     int cur_node,
                     T *heap,
                     int *size,
                     size_t node_size,
                     int *locks,
                     int lowest_level_start,
                     SharedMemoryLayout<T> shmem,
		     Compare const& compare) {

  int lane = g.thread_rank();
  int dim = g.size();

  int parent = Parent(cur_node, lowest_level_start);

  // Swim the new node up the tree
  while (cur_node != 1) {
    AcquireLock(g, &(locks[parent]));

    // If the heap property is already satisfied for this node and its
    // parent we are done
    if (!compare(heap[cur_node * node_size],
        heap[parent * node_size + node_size - 1])) {
      ReleaseLock(g, &(locks[parent]));
      break;
    }

    MergeAndSort(g, &heap[parent * node_size],
                   &heap[cur_node * node_size],
                   shmem.A,
                   shmem.B,
                   node_size,
                   shmem,
		   compare);

    g.sync();

    CopyPairs(g, &heap[parent * node_size], shmem.A, node_size);
    CopyPairs(g, &heap[cur_node * node_size], shmem.B, node_size);

    ReleaseLock(g, &(locks[cur_node]));
    cur_node = parent;
    parent = Parent(cur_node, lowest_level_start);
  }

  ReleaseLock(g, &(locks[cur_node]));
  
}

/**
* Sink the root down the heap 
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
*/
template <typename T, typename CG, typename Compare>
__device__ void Sink(CG const& g,
                     T *heap,
                     int *size,
                     size_t node_size,
                     int *locks,
                     size_t *p_buffer_size,
                     int lowest_level_start,
                     int node_capacity,
                     SharedMemoryLayout<T> shmem,
		     Compare const& compare) {

  size_t cur = kRootIdx;

  int dim = g.size();

  // Sink the node
  while (InsertionOrderIndex(LeftChild(cur, lowest_level_start),
         lowest_level_start) <= node_capacity) {

    size_t left = LeftChild(cur, lowest_level_start);
    size_t right = RightChild(cur, lowest_level_start);

    AcquireLock(g, &locks[left]);

    // The left node might have been removed
    // since the while loop condition, in which
    // case we are already at the bottom of the heap
    if (InsertionOrderIndex(left, lowest_level_start) > *size) {
      ReleaseLock(g, &locks[left]);
      break;
    }

    size_t lo;

    if (InsertionOrderIndex(right, lowest_level_start) <= node_capacity) {
      AcquireLock(g, &locks[right]);

      // Note that even with the bit reversal permutation,
      // we can never have a right child without a left child
      //
      // If we have both children, merge and sort them
      if (InsertionOrderIndex(right, lowest_level_start) <= *size) {

        size_t hi;

        // In order to ensure we preserve the heap property,
        // we put the largest node_size elements in the child
        // that previously contained the largest element
        if (!compare(heap[(left+1) * node_size - 1],
            heap[(right+1) * node_size - 1])) {
          hi = left;
          lo = right;
        } else {
          lo = left;
          hi = right;
        }

        // Skip the merge and sort if the nodes are already correctly
        // sorted
        if (!compare(heap[(lo+1) * node_size - 1],
            heap[hi * node_size])) {
          MergeAndSort(g, &heap[left * node_size],
                         &heap[right * node_size],
                         shmem.A,
                         shmem.B,
                         node_size,
                         shmem,
			 compare);

          g.sync();

          CopyPairs(g, &heap[hi * node_size], shmem.B, node_size);
          CopyPairs(g, &heap[lo * node_size], shmem.A, node_size);

          g.sync();
        }
        ReleaseLock(g, &locks[hi]);
      } else {
        lo = left;
        ReleaseLock(g, &locks[right]);
      }
    } else {
      lo = left;
    }

    // If the heap property is already satisfied between the current
    // node and the lower child, we are done return
    //
    // TODO: can this ever even occur? In the paper this is done because
    // a max placeholder value is used to indicate unused nodes in the heap
    if (!compare(heap[lo * node_size],
        heap[(cur + 1) * node_size - 1])) {
      ReleaseLock(g, &locks[lo]);
      ReleaseLock(g, &locks[cur]);
      return;
    }

    MergeAndSort(g, &heap[lo * node_size],
                   &heap[cur * node_size],
                   shmem.A,
                   shmem.B,
                   node_size,
                   shmem,
		   compare);

    g.sync();

    CopyPairs(g, &heap[lo * node_size], shmem.B, node_size);
    CopyPairs(g, &heap[cur * node_size], shmem.A, node_size);

    g.sync();

    ReleaseLock(g, &locks[cur]);

    cur = lo;

  }
  ReleaseLock(g, &locks[cur]);
  
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
*/
template <typename InputIt, typename T, typename Compare, typename CG>
__device__ void PushSingleNode(CG const& g,
                                 InputIt elements,
                                 T *heap,
                                 int *size,
                                 size_t node_size,
                                 int *locks,
                                 int lowest_level_start,
                                 SharedMemoryLayout<T> shmem,
				 Compare const& compare) {

  int lane = g.thread_rank();
  int dim = g.size();

  CopyPairs(g, shmem.A, elements, elements + node_size);

  g.sync();

  PBSort(g, shmem.A, node_size, node_size, shmem.B, compare);

  int *cur_node_temp = (int*)shmem.intersections;
  if (lane == 0) {
    *cur_node_temp = atomicAdd(size, 1) + 1;
  }
  g.sync();

  int cur_node = InsertionOrderIndex(*cur_node_temp, lowest_level_start);

  AcquireLock(g, &(locks[cur_node]));

  CopyPairs(g, &heap[cur_node * node_size], shmem.A, node_size);

  g.sync();

  Swim(g, cur_node, heap, size, node_size, locks,
            lowest_level_start, shmem, compare);

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
* @param pop_tracker The pop tracker for this concurrent pop operation
*                    (see PopKernel)
* @param lowest_level_start Index of the first node in the last level of the
*                           heap
* @param node_capacity Maximum capacity of the heap in nodes
* @param shmem The shared memory layout for this cooperative group
*/
template <typename OutputIt, typename T, typename Compare, typename CG>
__device__ void PopSingleNode(CG const& g,
                              OutputIt elements,
                              T *heap,
                              int *size,
                              size_t node_size,
                              int *locks,
                              size_t *p_buffer_size,
                              int *pop_tracker,
                              int lowest_level_start,
                              int node_capacity,
                              SharedMemoryLayout<T> shmem,
			      Compare const& compare) {

  int lane = g.thread_rank();
  int dim = g.size();

  AcquireLock(g, &locks[kRootIdx]);

  // Find the target node (the last one inserted) and
  // decrement the size

  size_t tar = InsertionOrderIndex(*size, lowest_level_start);

  if (tar != 1) {
    AcquireLock(g, &locks[tar]);
  }

  // pop_tracker determines our location in the output array,
  // since it tells us how many other nodes have been previously been
  // extracted by this block or by other blocks
  int out_idx = *pop_tracker;
  g.sync();

  if (lane == 0) {
    *size -= 1;
    *pop_tracker += 1;
  }
  g.sync();

  // Copy the root to the output array

  CopyPairs(g, elements + out_idx * node_size, &heap[node_size],
            &heap[node_size] + node_size);

  g.sync();

  // Copy the target node to the root

  if (tar != kRootIdx) {
    CopyPairs(g, &heap[node_size], &heap[tar * node_size],
              node_size);

    ReleaseLock(g, &locks[tar]);

    g.sync();
  }

  // Merge and sort the root and the partial buffer

  MergeAndSort(g, &heap[node_size],
                 &heap[kPBufferIdx],
                 shmem.A,
                 shmem.B,
                 node_size,
                 *p_buffer_size,
                 node_size,
                 shmem,
		 compare);

  g.sync();

  CopyPairs(g, &heap[node_size], shmem.A, node_size);

  CopyPairs(g, heap, shmem.B, *p_buffer_size);

  g.sync();

  Sink(g, heap, size, node_size, locks, p_buffer_size,
            lowest_level_start, node_capacity, shmem, compare);

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
*/
template <typename InputIt, typename T, typename Compare, typename CG>
__device__ void PopPartialNode(CG const& g,
                               InputIt elements,
                               size_t num_elements,
                               T *heap,
                               int *size,
                               size_t node_size,
                               int *locks,
                               size_t *p_buffer_size,
                               int lowest_level_start,
                               int node_capacity,
                               SharedMemoryLayout<T> shmem,
			       Compare const& compare) {
  int lane = g.thread_rank();
  int dim = g.size();

  AcquireLock(g, &locks[kRootIdx]);

  if (*size == 0) {
    CopyPairs(g, elements, heap, num_elements);
    g.sync();

    size_t n_p_buffer_size = *p_buffer_size - num_elements;

    if (n_p_buffer_size > 0) {
      size_t remaining = n_p_buffer_size;
      size_t index = 0;
      while (remaining > 0) {
        size_t this_round = min(remaining, num_elements);
        CopyPairs(g, heap + index, heap + index + num_elements,
                  this_round);
        remaining -= this_round;
        index += this_round;
        g.sync();
      }
    }

    if (lane == 0) {
      *p_buffer_size = n_p_buffer_size;
    }
    ReleaseLock(g, &locks[kRootIdx]);
  } else {

    CopyPairs(g, elements, &heap[kRootIdx * node_size], num_elements);
    g.sync();

    if (*p_buffer_size >= num_elements) {


      MergeAndSort(g, &heap[kPBufferIdx],
                      &heap[kRootIdx * node_size] + num_elements,
                      shmem.A,
                      shmem.B,
                      *p_buffer_size,
                      node_size - num_elements,
                      node_size,
                      shmem,
		      compare);

      if (lane == 0) {
        *p_buffer_size = *p_buffer_size - num_elements;
      }

      g.sync();

      CopyPairs(g, &heap[kRootIdx * node_size], shmem.A, node_size);
      CopyPairs(g, &heap[kPBufferIdx], shmem.B, *p_buffer_size);

      g.sync();

      Sink(g, heap, size, node_size, locks, p_buffer_size,
                lowest_level_start, node_capacity, shmem, compare);
    } else {

      MergeAndSort(g, &heap[kPBufferIdx],
                      &heap[kRootIdx * node_size] + num_elements,
                      shmem.A,
                      (T*)nullptr,
                      *p_buffer_size,
                      node_size - num_elements,
                      node_size,
                      shmem,
		      compare);

      g.sync();

      CopyPairs(g, &heap[kPBufferIdx], shmem.A,
                *p_buffer_size + node_size - num_elements);

      int tar = InsertionOrderIndex(*size, lowest_level_start);
      g.sync();

      if (lane == 0) {
        *size -= 1;
      }

      if (tar != kRootIdx) {
        AcquireLock(g, &locks[tar]);

        CopyPairs(g, &heap[kRootIdx * node_size],
                  &heap[tar * node_size], node_size);

        g.sync();

        ReleaseLock(g, &locks[tar]);

        MergeAndSort(g, &heap[node_size],
                       &heap[kPBufferIdx],
                       shmem.A,
                       shmem.B,
                       node_size,
                       *p_buffer_size,
                       node_size,
                       shmem,
		       compare);

        g.sync();

        CopyPairs(g, &heap[node_size], shmem.A, node_size);

        CopyPairs(g, heap, shmem.B, *p_buffer_size);

        g.sync();

        Sink(g, heap, size, node_size, locks,
                  p_buffer_size, lowest_level_start, node_capacity, shmem,
		  compare);
      } else {
        ReleaseLock(g, &locks[kRootIdx]);
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
*/
template <typename InputIt, typename T, typename Compare, typename CG>
__device__ void PushPartialNode(CG const& g,
                                InputIt elements,
                                size_t p_ins_size,
                                T *heap,
                                int *size,
                                size_t node_size,
                                int *locks,
                                size_t *p_buffer_size, 
                                int lowest_level_start,
                                SharedMemoryLayout<T> shmem,
				Compare const& compare) {

  int lane = g.thread_rank();
  int dim = g.size();

  AcquireLock(g, &locks[kRootIdx]);

  CopyPairs(g, shmem.B, elements, p_ins_size);

  PBSort(g, shmem.B, p_ins_size, node_size, shmem.A, compare);

  // There is enough data for a new node, in which case we
  // construct a new node and insert it
  if (*p_buffer_size + p_ins_size >= node_size) {

    int *cur_node_temp = shmem.intersections;
    if (lane == 0) {
      *cur_node_temp = atomicAdd(size, 1) + 1;
    }
    g.sync();

    int cur_node = InsertionOrderIndex(*cur_node_temp, lowest_level_start);

    if (cur_node != kRootIdx) {
      AcquireLock(g, &(locks[cur_node]));
    }

    g.sync();

    MergeAndSort(g, shmem.B,
                   &heap[kPBufferIdx],
                   &heap[cur_node * node_size],
                   shmem.A,
                   p_ins_size,
                   *p_buffer_size,
                   node_size,
                   shmem,
		   compare);

    if (lane == 0) {
      *p_buffer_size = (*p_buffer_size + p_ins_size) - node_size;
    }

    g.sync();

    CopyPairs(g, heap, shmem.A, *p_buffer_size);

    if (cur_node != kRootIdx) {
      ReleaseLock(g, &locks[kRootIdx]);
    }

    Swim(g, cur_node, heap, size, node_size,
              locks, lowest_level_start, shmem, compare);

  } else {
    // There are not enough elements for a new node,
    // in which case we merge and sort the root and
    // the elements to be inserted and then the root
    // and the partial buffer

    MergeAndSort(g, shmem.B,
                   &heap[kPBufferIdx],
                   shmem.A,
                   (T*)nullptr,
                   p_ins_size,
                   *p_buffer_size,
                   node_size,
                   shmem,
		   compare);

    g.sync();

    if (lane == 0) {
      *p_buffer_size += p_ins_size;
    }

    g.sync();

    CopyPairs(g, heap, shmem.A, *p_buffer_size);

    g.sync();

    if (*size > 0) {
      MergeAndSort(g, &heap[node_size],
                     &heap[kPBufferIdx],
                     shmem.A,
                     shmem.B,
                     node_size,
                     *p_buffer_size,
                     node_size,
                     shmem,
		     compare);
      g.sync();

      CopyPairs(g, heap, shmem.B, *p_buffer_size);

      CopyPairs(g, &heap[node_size], shmem.A, node_size);
    }
    ReleaseLock(g, &locks[kRootIdx]);
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
*/
template <typename OutputIt, typename T, typename Compare>
__global__ void PushKernel(OutputIt elements,
                           size_t num_elements,
                           T *heap,
                           int *size,
                           size_t node_size,
                           int *locks,
                           size_t *p_buffer_size,
                           int lowest_level_start,
			   Compare compare) {

  extern __shared__ int s[];

  SharedMemoryLayout<T> shmem = GetSharedMemoryLayout<T>(s,
                                              blockDim.x, node_size);

  // We push as many elements as possible as full nodes,
  // then deal with the remaining elements as a partial insertion
  // below
  thread_block g = this_thread_block();
  for (size_t i = blockIdx.x * node_size;
       i + node_size <= num_elements;
       i += gridDim.x * node_size) {
    PushSingleNode(g, elements + i, heap, size, node_size, locks,
                   lowest_level_start, shmem, compare);
  }

  // We only need one block for partial insertion
  if (blockIdx.x != 0) {
    return;
  }

  // If node_size does not divide num_elements, there are some leftover
  // elements for which we must perform a partial insertion
  size_t first_not_inserted = (num_elements / node_size)
                                             * node_size;

  if (first_not_inserted < num_elements) {
    size_t p_ins_size = num_elements - first_not_inserted;
    PushPartialNode(g, elements + first_not_inserted, p_ins_size,
                         heap, size, node_size, locks, p_buffer_size,
                         lowest_level_start, shmem, compare);
  }
}

/**
* Add num_elements elements into the heap from
* elements, using a warp to handle each node rather than a block
* @param elements The array of elements to add
* @param num_elements The number of elements to be inserted
* @param heap The array of pairs that stores the heap itself
* @param size Pointer to the number of pairs currently in the heap
* @param node_size Size of the nodes in the heap
* @param locks Array of locks, one for each node in the heap
* @param p_buffer_size Number of pairs in the heap's partial buffer
* @param temp_node A temporary array large enough to store
                   sizeof(T) * node_size bytes
*/
template <typename InputIt, typename T, typename Compare>
__global__ void PushKernelWarp(InputIt elements,
                           size_t num_elements,
                           T *heap,
                           int *size,
                           size_t node_size,
                           int *locks,
                           size_t *p_buffer_size,
                           int lowest_level_start,
                           int bytes_shmem_per_warp,
			   Compare compare) {

  extern __shared__ char sh[];

  // We push as many elements as possible as full nodes,
  // then deal with the remaining elements as a partial insertion
  // below
  thread_block block = this_thread_block();
  thread_block_tile<32> warp = tiled_partition<32>(block); 

  SharedMemoryLayout<T> shmem = GetSharedMemoryLayout<T>(
                  (int*)(sh + bytes_shmem_per_warp * warp.meta_group_rank()),
                         32, node_size);

  for (size_t i = warp.meta_group_rank() * node_size
                   + blockIdx.x * node_size * (blockDim.x / 32);
       i + node_size <= num_elements;
       i += (blockDim.x / 32) * node_size * gridDim.x) {
    PushSingleNode(warp, elements + i, heap, size, node_size, locks,
                   lowest_level_start, shmem, compare);
  }

  // We only need one block for partial insertion
  if (blockIdx.x != 0 || warp.meta_group_rank() != 0) {
    return;
  }

  // If node_size does not divide num_elements, there are some leftover
  // elements for which we must perform a partial insertion
  size_t first_not_inserted = (num_elements / node_size)
                                             * node_size;

  if (first_not_inserted < num_elements) {
    size_t p_ins_size = num_elements - first_not_inserted;
    PushPartialNode(warp, elements + first_not_inserted, p_ins_size,
                         heap, size, node_size, locks, p_buffer_size,
                         lowest_level_start, shmem, compare);
  }
}

/**
* Remove exactly node_size elements from the heap and place them
* in elements, using a warp to handle each node rather than a block
* @param elements The array of elements to insert into
* @param num_elements The number of elements to remove
* @param heap The array of pairs that stores the heap itself
* @param size Pointer to the number of pairs currently in the heap
* @param node_size Size of the nodes in the heap
* @param locks Array of locks, one for each node in the heap
* @param p_buffer_size Number of pairs in the heap's partial buffer
* @param pop_tracker Pointer to an integer in global memory initialized to 0
*/
template <typename OutputIt, typename T, typename Compare>
__global__ void PopKernelWarp(OutputIt elements,
                           size_t num_elements,
                           T *heap,
                           int *size,
                           size_t node_size,
                           int *locks,
                           size_t *p_buffer_size,
                           int *pop_tracker,
                           int lowest_level_start,
                           int node_capacity,
                           int bytes_shmem_per_warp,
			   Compare compare) {

  // We use pop_tracker to ensure that each thread block inserts its node
  // at the correct location in the output array
  // Since we do not know which block will extract which node

  extern __shared__ char sh[];

  thread_block block = this_thread_block();
  thread_block_tile<32> warp = tiled_partition<32>(block); 

  SharedMemoryLayout<T> shmem = GetSharedMemoryLayout<T>(
                   (int*)(sh + bytes_shmem_per_warp * warp.meta_group_rank()), 
                          32, node_size);

  for (size_t i = warp.meta_group_rank() + (blockDim.x / 32) * blockIdx.x;
       i < num_elements / node_size;
       i += gridDim.x * blockDim.x / 32) {
    PopSingleNode(warp, elements, heap, size, node_size, locks,
                       p_buffer_size, pop_tracker, lowest_level_start,
                       node_capacity, shmem, compare);
  }

  AcquireLock(warp, &locks[kRootIdx]);
  // Remove from the partial buffer if there are no nodes
  // Only one thread will attempt this deletion because we have acquired
  // the root and will increment pop_tracker once we begin the deletion
  if (*pop_tracker == num_elements / node_size
      && num_elements % node_size != 0) {

    if (warp.thread_rank() == 0) {
      *pop_tracker += 1;
    }

    size_t p_del_size = num_elements % node_size;

    ReleaseLock(warp, &locks[kRootIdx]);

    PopPartialNode(warp, 
                   elements + (num_elements / node_size) * node_size,
                   p_del_size, heap, size, node_size, locks, p_buffer_size,
                   lowest_level_start, node_capacity, shmem, compare);
    
  } else {
    ReleaseLock(warp, &locks[kRootIdx]);
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
* @param pop_tracker Pointer to an integer in global memory initialized to 0
*/
template <typename OutputIt, typename T, typename Compare>
__global__ void PopKernel(OutputIt elements,
                           size_t num_elements,
                           T *heap,
                           int *size,
                           size_t node_size,
                           int *locks,
                           size_t *p_buffer_size,
                           int *pop_tracker,
                           int lowest_level_start,
                           int node_capacity,
			   Compare compare) {

  // We use pop_tracker to ensure that each thread block inserts its node
  // at the correct location in the output array
  // Since we do not know which block will extract which node

  extern __shared__ int s[];

  SharedMemoryLayout<T> shmem = GetSharedMemoryLayout<T>(s,
                                                       blockDim.x, node_size);

  thread_block g = this_thread_block();
  for (size_t i = blockIdx.x; i < num_elements / node_size; i += gridDim.x) {
    PopSingleNode(g, elements, heap, size, node_size, locks,
                       p_buffer_size, pop_tracker, lowest_level_start,
                       node_capacity, shmem, compare);
  }

  AcquireLock(g, &locks[kRootIdx]);
  // Remove from the partial buffer if there are no nodes
  // Only one thread will attempt this deletion because we have acquired
  // the root and will increment pop_tracker once we begin the deletion
  if (*pop_tracker == num_elements / node_size
      && num_elements % node_size != 0) {

    if (g.thread_rank() == 0) {
      *pop_tracker += 1;
    }

    size_t p_del_size = num_elements % node_size;

    ReleaseLock(g, &locks[kRootIdx]);

    PopPartialNode(g, elements + (num_elements / node_size) * node_size,
                   p_del_size, heap, size, node_size, locks, p_buffer_size,
                   lowest_level_start, node_capacity, shmem, compare);
    
  } else {
    ReleaseLock(g, &locks[kRootIdx]);
  }
}
}
