#pragma once

#include <vector>
#include <utility>
#include <cuco/allocator.hpp>

#include <thrust/functional.h>

namespace cuco {

/*
* @brief A GPU-accelerated priority queue of key-value pairs
*
* Allows for multiple concurrent insertions as well as multiple concurrent
* deletions
*
* Current limitations:
* - Only supports trivially comparable key types
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
* The device-side operations allow a cooperative group to push or pop
* some number of elements less than or equal to node_size. These device side
* operations are invoked with a trivially-copyable device view,
* `device_mutable_view` which can be obtained with the host function 
* `get_mutable_device_view` and passed to the device.
*
* @tparam Key Trivially comparable type used for keys
* @tparam Value Type of the value to be stored
* @tparam Max When false, pop operations yield the elements with the smallest
*             keys in the queue, otherwise, pop operations yeild the elements
*             with the largest keys
*/
template <typename T, typename Compare = thrust::less<T>,
	  bool FavorInsertionPerformance = false,
	  typename Allocator = cuco::cuda_allocator<char>>
class priority_queue {

  using int_allocator_type = typename std::allocator_traits<Allocator>
	                        ::rebind_alloc<int>;

  using t_allocator_type = typename std::allocator_traits<Allocator>
	                                  ::rebind_alloc<T>;
  
  using size_t_allocator_type = typename std::allocator_traits<Allocator>
	                                  ::rebind_alloc<size_t>;

  const int NodeSize = FavorInsertionPerformance ? 64 : 1024;

 public:
  /**
   * @brief Construct a priority queue
   *
   * @param initial_capacity The number of elements the priority queue can hold
   * @param node_size The size of the nodes in the underlying heap data
   *        structure
   */
  priority_queue(size_t initial_capacity, Allocator const& alloc = Allocator{});

  /**
   * @brief Push elements into the priority queue
   *
   * @tparam InputIt Device accessible input iterator whose `value_type`
   *        can be converted to T
   * @param first Beginning of the sequence of elements
   * @param last End of the sequence of elements
   * @param num_elements Number of elements to add to the queue
   * @param block_size Block size to use for the internal kernel launch
   * @param grid_size Grid size for the internal kernel launch
   * @param warp_size If true, each node is handled by a single warp, otherwise
   *                  by a single block
   * @param stream The stream in which the underlying GPU operations will be
   *               run
   */
  template <typename InputIt>
  void push(InputIt first, InputIt last, cudaStream_t stream = 0);

  /**
   * @brief Remove a sequence of the lowest (when Max == false) or the
   *        highest (when Max == true) elements
   *
   * @tparam OutputIt Device accessible output iterator whose `value_type`
   *        can be converted to T
   * @param first Beginning of the sequence of output elements
   * @param last End of the sequence of output elements
   * @param num_elements The number of elements to be removed
   * @param block_size Block size to use for the internal kernel launch
   * @param grid_size Grid size for the internal kernel launch
   * @param warp_size If true, each node is handled by a single warp, otherwise
   *                  by a single block
   * @param stream The stream in which the underlying GPU operations will be
   *               run
   */
  template <typename OutputIt>
  void pop(OutputIt first, OutputIt last, cudaStream_t stream = 0);

  /*
  * @brief Return the amount of shared memory required for operations on the queue
  * with a thread block size of block_size
  *
  * @param block_size Size of the blocks to calculate storage for
  * @return The amount of temporary storage required in bytes
  */
  int get_shmem_size(int block_size) {
    int intersection_bytes = 2 * (block_size + 1) * sizeof(int);
    int node_bytes = node_size_ * sizeof(T);
    return intersection_bytes + 2 * node_bytes;
  }

  /**
   * @brief Destroys the queue and frees its contents
   */
  ~priority_queue();

  class device_mutable_view {
   public:

    /**
     * @brief Push a single node or less elements into the priority queue
     *
     * @tparam CG Cooperative Group type
     * @tparam Device accessible iterator whose `value_type` is convertible
     *         to T
     * @param g The cooperative group that will perform the operation
     * @param first The beginning of the sequence of elements to insert
     * @param last The end of the sequence of elements to insert
     * @param Pointer to a contiguous section of memory large enough
     *        to hold get_shmem_size(g.size()) bytes
     */
    template <typename CG, typename InputIt>
    __device__ void push(CG const& g, InputIt first,
                         InputIt last, void *temp_storage);

    /**
     * @brief Pop a single node or less elements from the priority queue
     *
     * @tparam CG Cooperative Group type
     * @tparam Device accessible iterator whose `value_type` is convertible to
               T
     * @param g The cooperative group that will perform the operation
     * @param first The beginning of the sequence of elements to output into
     * @param last The end of the sequence of elements to output into
     * @param Pointer to a contiguous section of memory large enough
     *        to hold get_shmem_size(g.size()) bytes
     */
    template <typename CG, typename OutputIt>
    __device__ void pop(CG const& g, OutputIt first,
                        OutputIt last, void *temp_storage);

    /**
     * @brief Returns the node size of the queue's underlying heap
     *        representation, i.e. the maximum number of elements
     *        pushable or poppable with a call to the device push
     *        and pop functions
     *
     * @return The underlying node size
     */
    __device__ size_t get_node_size() {
      return node_size_;
    }

    /*
    * @brief Return the amount of temporary storage required for operations
    * on the queue with a cooperative group size of block_size
    *
    * @param block_size Size of the cooperative groups to calculate storage for
    * @return The amount of temporary storage required in bytes
    */
    __device__ int get_shmem_size(int block_size) {
      int intersection_bytes = 2 * (block_size + 1) * sizeof(int);
      int node_bytes = node_size_ * sizeof(T);
      return intersection_bytes + 2 * node_bytes;
    }

    __host__ __device__ device_mutable_view(size_t node_size,
                                            T *d_heap,
                                            int *d_size,
                                            size_t *d_p_buffer_size,
                                            int *d_locks,
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
    size_t node_size_;
    int lowest_level_start_;
    int node_capacity_;

    T *d_heap_;
    int *d_size_;
    size_t *d_p_buffer_size_;
    int *d_locks_;
    Compare compare_;
  };

  /*
  * @brief Returns a trivially-copyable class that can be used to perform 
  *        insertion and deletion of single nodes in device code with
  *        cooperative groups
  *
  * @return A device view
  */
  device_mutable_view get_mutable_device_view() {
    return device_mutable_view(node_size_, d_heap_, d_size_, d_p_buffer_size_,
                               d_locks_, lowest_level_start_, node_capacity_,
			       compare_);
  }

 private:
  size_t node_size_;         ///< Size of the heap's nodes
  int lowest_level_start_;   ///< Index in `d_heap_` of the first node in the
                             ///  heap's lowest level
  int node_capacity_;        ///< Capacity of the heap in nodes

  T *d_heap_; ///< Pointer to an array of nodes, the 0th node
                             ///  being the heap's partial buffer, and nodes
                             ///  1..(node_capacity_) being the heap, where the
                             ///  1st node is the root
  int *d_size_;              ///< Number of nodes currently in the heap
  size_t *d_p_buffer_size_;  ///< Number of elements currently in the partial
                             ///  buffer
  int *d_locks_;             ///< Array of locks where `d_locks_[i]` is the
                             ///  lock for the node starting at
                             ///  1d_heap_[node_size * i]`

  Allocator allocator_;
  int_allocator_type int_allocator_;
  t_allocator_type t_allocator_;
  size_t_allocator_type size_t_allocator_;

  Compare compare_{};
};

}

#include <cuco/detail/priority_queue.inl>
