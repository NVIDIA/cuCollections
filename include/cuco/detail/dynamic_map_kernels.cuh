// insert kernel
template<uint32_t block_size,
         typename pair_type,
         typename InputIt,
         typename viewT,
         typename mutableViewT,
         typename atomicT,
         typename Hash, 
         typename KeyEqual>
__global__ void insertKernel(InputIt first,
                             InputIt last,
                             viewT* submap_views,
                             mutableViewT* submap_mutable_views,
                             atomicT* num_successes,
                             uint32_t insert_idx,
                             Hash hash,
                             KeyEqual key_equal) {
  
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;
  
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto empty_key_sentinel = submap_views[0].get_empty_key_sentinel();
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();

  while(first + tid < last) {
    pair_type insert_pair = first[tid];
    auto exists = false;
    for(auto i = 0; i < insert_idx; ++i) {
      auto submap_view = submap_views[i];
      exists = submap_view.contains(insert_pair.first, hash, key_equal);
      if(exists) {
        break;
      }
    }
    if(!exists) {
      auto res = submap_mutable_views[insert_idx].insert(insert_pair, hash, key_equal);
      if(res.second) {
        thread_num_successes++;
      }
    }

    tid += gridDim.x * blockDim.x;
  }
  
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if(threadIdx.x == 0) {
    *num_successes += block_num_successes;
  }
}



// find kernel
template<typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void findKernel(InputIt first,
                           InputIt last,
                           OutputIt output_begin,
                           viewT* submap_views,
                           uint32_t num_submaps,
                           Hash hash,
                           KeyEqual key_equal) {
  
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto empty_key_sentinel = submap_views[0].get_empty_key_sentinel();                          
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();                          

  while(first + tid < last) {
    auto key = first[tid];
    auto found_value = empty_value_sentinel;
    for(auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.find(key, hash, key_equal);
      if(found->first != empty_key_sentinel) {
        found_value = found->second;
        break;
      }
    }

    output_begin[tid] = found_value;
    tid += gridDim.x * blockDim.x;
  }
}



// contains kernel
template<typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void containsKernel(InputIt first,
                               InputIt last,
                               OutputIt output_begin,
                               viewT* submap_views,
                               uint32_t num_submaps,
                               Hash hash,
                               KeyEqual key_equal) {
  
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto empty_key_sentinel = submap_views[0].get_empty_key_sentinel();                          
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();                          

  while(first + tid < last) {
    auto key = first[tid];
    auto found = false;
    for(auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.contains(key, hash, key_equal);
      if(found) {
        break;
      }
    }

    output_begin[tid] = found;
    tid += gridDim.x * blockDim.x;
  }
}