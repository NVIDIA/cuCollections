#pragma once

#include <vector>
#include <utility>
#include <cuco/detail/pq_pair.cuh>

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
template <typename Key, typename Value, bool Max = false>
class priority_queue {

 public:
  /**
   * Construct a priority queue
   * @param initial_capacity The number of elements the priority queue can hold
   * @param node_size The size of the nodes in the underlying heap data
   *        structure
   */
  priority_queue(size_t initial_capacity, size_t node_size = 1024);

  /**
   * Push num_elements elements into the priority queue
   * @param elements Array of elements to add to the queue
   * @param num_elements Number of elements to add to the queue
   * @param block_size Block size to use for the internal kernel launch
   * @param grid_size Grid size for the internal kernel launch
   * @param warp_size If true, each node is handled by a single warp, otherwise
   *                  by a single block
   * @param stream The stream in which the underlying GPU operations will be
   *               run
   */
  void push(Pair<Key, Value> *elements, size_t num_elements,
            int block_size = 256, int grid_size = 64000,
            bool warp_level = false,
            cudaStream_t stream = 0);

  /**
   * Remove the num_elements elements with the lowest keys from the priority
   * queue and place them in out in ascending sorted order by key
   * @param out The array in which the removed elements will be placed
   * @param num_elements The number of elements to be removed
   * @param block_size Block size to use for the internal kernel launch
   * @param grid_size Grid size for the internal kernel launch
   * @param warp_size If true, each node is handled by a single warp, otherwise
   *                  by a single block
   * @param stream The stream in which the underlying GPU operations will be
   *               run
   */
  void pop(Pair<Key, Value> *out, size_t num_elements,
           int block_size = 512, int grid_size = 32000,
           bool warp_level = false,
           cudaStream_t stream = 0);

  /*
  * Return the amount of shared memory required for operations on the queue
  * with a thread block size of block_size
  *
  * @param block_size Size of the blocks to calculate storage for
  */
  int get_shmem_size(int block_size) {
    int intersection_bytes = 2 * (block_size + 1) * sizeof(int);
    int node_bytes = node_size_ * sizeof(Pair<Key, Value>);
    return intersection_bytes + 2 * node_bytes;
  }

  ~priority_queue();

  class device_mutable_view {
   public:

    /**
     * Push a single node or less elements into the priority queue
     *
     * @param g The cooperative group that will perform the operation
     * @param elements Array of elements to add to the queue
     * @param num_elements Number of elements to add to the queue
     * @param Pointer to a contiguous section of memory large enough
     *        to hold get_shmem_size(g.size()) bytes
     */
    template <typename CG>
    __device__ void push(CG const& g, Pair<Key, Value> *elements,
                         size_t num_elements, void *temp_storage);

    /**
     * Pop a single node or less elements from the priority queue
     *
     * @param g The cooperative group that will perform the operation
     * @param out Array of elements to put the removed elements in 
     * @param num_elements Number of elements to remove from the queue
     * @param Pointer to a contiguous section of memory large enough
     *        to hold get_shmem_size(g.size()) bytes
     */
    template <typename CG>
    __device__ void pop(CG const& g, Pair<Key, Value> *out,
                        size_t num_elements, void *temp_storage);

    __device__ size_t get_node_size() {
      return node_size_;
    }

    /*
    * Return the amount of temporary storage required for operations
    * on the queue with a cooperative group size of block_size
    *
    * @param block_size Size of the cooperative groups to calculate storage for
    */
    __device__ int get_shmem_size(int block_size) {
      int intersection_bytes = 2 * (block_size + 1) * sizeof(int);
      int node_bytes = node_size_ * sizeof(Pair<Key, Value>);
      return intersection_bytes + 2 * node_bytes;
    }

    __host__ __device__ device_mutable_view(size_t node_size,
                                            Pair<Key, Value> *d_heap,
                                            int *d_size,
                                            size_t *d_p_buffer_size,
                                            int *d_locks,
                                            int lowest_level_start,
                                            int node_capacity)
      : node_size_(node_size),
        d_heap_(d_heap),
        d_size_(d_size),
        d_p_buffer_size_(d_p_buffer_size),
        d_locks_(d_locks),
        lowest_level_start_(lowest_level_start),
        node_capacity_(node_capacity)
    {
    }

   private:
    size_t node_size_;
    int lowest_level_start_;
    int node_capacity_;

    Pair<Key, Value> *d_heap_;
    int *d_size_;
    size_t *d_p_buffer_size_;
    int *d_locks_;
  };

  /*
  * Return a class that can be used to perform insertion and deletion
  * of single nodes in device code with cooperative groups
  */
  device_mutable_view get_mutable_device_view() {
    return device_mutable_view(node_size_, d_heap_, d_size_, d_p_buffer_size_,
                               d_locks_, lowest_level_start_, node_capacity_);
  }

 private:
  size_t node_size_;
  int lowest_level_start_;
  int node_capacity_;

  Pair<Key, Value> *d_heap_;
  int *d_size_;
  size_t *d_p_buffer_size_;
  int *d_locks_;
  int *d_pop_tracker_;
};

}

#include <cuco/detail/priority_queue.inl>
