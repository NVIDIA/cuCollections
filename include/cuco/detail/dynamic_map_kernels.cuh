namespace detail {
namespace cg = cooperative_groups;

// insert kernel
template<uint32_t block_size,
         typename pair_type,
         typename InputIt,
         typename viewT,
         typename mutableViewT,
         typename atomicT,
         typename Hash, 
         typename KeyEqual>
__global__ void insert(InputIt first,
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



// insert kernel
template<uint32_t block_size,
         uint32_t tile_size,
         typename pair_type,
         typename InputIt,
         typename viewT,
         typename mutableViewT,
         typename atomicT,
         typename Hash, 
         typename KeyEqual>
__global__ void insert(InputIt first,
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
  
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it = first + tid / tile_size;
  auto empty_key_sentinel = submap_views[0].get_empty_key_sentinel();
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();

  while(it < last) {
    pair_type insert_pair = *it;
    auto exists = false;
    
    for(auto i = 0; i < insert_idx; ++i) {
      auto submap_view = submap_views[i];
      exists = submap_view.contains(tile, insert_pair.first, hash, key_equal);
      if(exists) {
        break;
      }
    }
    if(!exists) {
      auto res = submap_mutable_views[insert_idx].insert(tile, insert_pair, hash, key_equal);
      if(tile.thread_rank() == 0 && res.second) {
        thread_num_successes++;
      }
    }

    it += (gridDim.x * blockDim.x) / tile_size;
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
__global__ void find(InputIt first,
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
      if(found != submap_view.end()) {
        found_value = found->second;
        break;
      }
    }

    output_begin[tid] = found_value;
    tid += gridDim.x * blockDim.x;
  }
}



// find kernel
template<uint32_t tile_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void find(InputIt first,
                     InputIt last,
                     OutputIt output_begin,
                     viewT* submap_views,
                     uint32_t num_submaps,
                     Hash hash,
                     KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  auto empty_key_sentinel = submap_views[0].get_empty_key_sentinel();                          
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();                          

  while(first + key_idx < last) {
    auto key = first[key_idx];
    auto found_value = empty_value_sentinel;
    for(auto i = 0; i < num_submaps; ++i) {
      auto submap_view = submap_views[i];
      auto found = submap_view.find(tile, key, hash, key_equal);
      if(found != submap_view.end()) {
        found_value = found->second;
        break;
      }
    }

    if(tile.thread_rank() == 0) {
      output_begin[key_idx] = found_value;
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}



// contains kernel
template<typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void contains(InputIt first,
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
      found = submap_views[i].contains(key, hash, key_equal);
      if(found) {
        break;
      }
    }

    output_begin[tid] = found;
    tid += gridDim.x * blockDim.x;
  }
}



// contains kernel
template<uint32_t tile_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void contains(InputIt first,
                         InputIt last,
                         OutputIt output_begin,
                         viewT* submap_views,
                         uint32_t num_submaps,
                         Hash hash,
                         KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  auto empty_key_sentinel = submap_views[0].get_empty_key_sentinel();                          
  auto empty_value_sentinel = submap_views[0].get_empty_value_sentinel();                          

  while(first + key_idx < last) {
    auto key = first[key_idx];
    auto found = false;
    for(auto i = 0; i < num_submaps; ++i) {
      found = submap_views[i].contains(tile, key, hash, key_equal);
      if(found) {
        break;
      }
    }

    if(tile.thread_rank() == 0) {
      output_begin[key_idx] = found;
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}
} // namespace detail