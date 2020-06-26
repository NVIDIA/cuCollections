template <typename KeyT, typename ValueT>
__device__ __forceinline__ bool
insertPairUnique(
    bool& mySuccess,
    bool& to_be_inserted,
    const uint32_t& laneId,
    const KeyT& myKey,
    const ValueT& myValue,
    const uint32_t bucket_id,
    AllocatorContextT& local_allocator_ctx) {
  uint32_t work_queue = 0;
  uint32_t last_work_queue = 0;
  uint32_t next = SlabHashT::A_INDEX_POINTER;
  bool new_insertion = false;
  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_inserted))) {
    // to know whether it is a base node, or a regular node
    next = (last_work_queue != work_queue) ? SlabHashT::A_INDEX_POINTER
                                           : next;  // a successful insertion in the warp
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_slot = __shfl_sync(0xFFFFFFFF, key_idx, src_lane, 32);

    uint32_t src_unit_data = getDataFromSubmap(src_bucket, laneId);
    uint64_t old_key_value_pair = 0;

    uint32_t isEmpty = (__ballot_sync(0xFFFFFFFF, src_unit_data == EMPTY_KEY)) &
                       SlabHashT::REGULAR_NODE_KEY_MASK;

    uint32_t src_key = __shfl_sync(0xFFFFFFFF, myKey, src_lane, 32);
    uint32_t isExisting = (__ballot_sync(0xFFFFFFFF, src_unit_data == src_key)) &
                          SlabHashT::REGULAR_NODE_KEY_MASK;
    if (isExisting) {  // key exist in the hash table
      if (laneId == src_lane) {
        mySuccess = true;
        to_be_inserted = false;
      }
    } else {
      if (isEmpty == 0) {  // no empty slot available:
        uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
        if (next_ptr == SlabHashT::EMPTY_INDEX_POINTER) {
          // allocate a new node:
          uint32_t new_node_ptr = allocateSlab(local_allocator_ctx, laneId);
          if(new_node_ptr == 0xFFFFFFFF) { // could not allocate a new slab: pool size needs to be increased
            mySuccess = false; // signal that this key needs to be reinserted 
            to_be_inserted = false;
            continue;
          }

          if (laneId == 31) {
            const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                                    ? getPointerFromBucket(src_bucket, 31)
                                    : getPointerFromSlab(next, 31);

            uint32_t temp =
                atomicCAS((unsigned int*)p, SlabHashT::EMPTY_INDEX_POINTER, new_node_ptr);
            // check whether it was successful, and
            // free the allocated memory otherwise
            if (temp != SlabHashT::EMPTY_INDEX_POINTER) {
              freeSlab(new_node_ptr);
            }
          }
        } else {
          next = next_ptr;
        }
      } else {  // there is an empty slot available
        int dest_lane = __ffs(isEmpty & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
        if (laneId == src_lane) {
          const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                                  ? getPointerFromBucket(src_bucket, dest_lane)
                                  : getPointerFromSlab(next, dest_lane);

          old_key_value_pair =
              atomicCAS((unsigned long long int*)p,
                        EMPTY_PAIR_64,
                        ((uint64_t)(*reinterpret_cast<const uint32_t*>(
                             reinterpret_cast<const unsigned char*>(&myValue)))
                         << 32) |
                            *reinterpret_cast<const uint32_t*>(
                                reinterpret_cast<const unsigned char*>(&myKey)));
          if (old_key_value_pair == EMPTY_PAIR_64) {
            mySuccess = true;
            to_be_inserted = false;  // successful insertion
            new_insertion = true;
          }
        }
      }
    }
    last_work_queue = work_queue;
  }
  return new_insertion;
}