namespace cg = cooperative_groups;

// insert kernel
template<typename key_type, typename mapped_type, typename value_type, typename viewT>
__global__ void insertKeySet(thrust::device_ptr<key_type> keys, 
               thrust::device_ptr<mapped_type> values,
               unsigned long long int *totalNumSuccesses,
               uint64_t numKeys, uint32_t submapIdx, viewT view) { 
  constexpr uint32_t BLOCK_SIZE = 128;
  constexpr uint32_t TILE_SIZE = BLOCK_SIZE * view.insertGran;
  uint64_t tid = TILE_SIZE * blockIdx.x + threadIdx.x;

  typedef cub::BlockReduce<uint64_t, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  
  uint64_t numSuccess = 0;
  for(auto i = 0; i < TILE_SIZE; i += BLOCK_SIZE) {
    auto idx = tid + i;
    if(idx >= numKeys) {
      break;
    }
    auto key = keys[idx];
    auto value = values[idx];
    value_type insertPair = thrust::make_pair(key, value);
    auto found = view.DupCheck(key, submapIdx);
    if(!found) {
      auto res = view.insert(submapIdx, insertPair);
      if(res.second) {
        numSuccess++;
      }
    }
  }

  // tally up number of successful insertions
  uint64_t blockNumSuccesses = BlockReduce(tempStorage).Sum(numSuccess);
  if(threadIdx.x == 0) {
    atomicAdd(totalNumSuccesses, static_cast<unsigned long long int>(blockNumSuccesses));
  }
}



// cooperative group based search kernel
template<uint32_t tileSize, typename value_type, typename key_type, 
         typename mapped_type, typename viewT>
__global__ void insertKeySetCG(thrust::device_ptr<key_type> keys, 
                               thrust::device_ptr<mapped_type> values,
                               unsigned long long int* totalNumSuccesses,
                               uint64_t numKeys, uint32_t submapIdx,
                               viewT view) {
  auto tile = cg::tiled_partition<tileSize>(cg::this_thread_block());
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  constexpr uint32_t BLOCK_SIZE = 128;
  typedef cub::BlockReduce<uint64_t, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t numSuccesses = 0;

  if(tid < tileSize * numKeys) {
    uint64_t keyIdx = tid / tileSize;
    auto key = keys[keyIdx];
    auto value = values[keyIdx];
    auto insertPair = thrust::make_pair(key, value);
    auto found = view.dupCheckCG(tile, key, submapIdx);
    if(!found) {
      auto res = view.insertCG(tile, insertPair, submapIdx);
      if(tile.thread_rank() == 0 && res.second) {
        numSuccesses = 1;
      }
    }
  }

  uint64_t blockNumSuccesses = BlockReduce(tempStorage).Sum(numSuccesses);
  if(threadIdx.x == 0) {
    atomicAdd(totalNumSuccesses, static_cast<unsigned long long int>(blockNumSuccesses));
  }
}



// search kernel
template<typename key_type, typename mapped_type, typename viewT>
__global__ void searchKeySet(thrust::device_ptr<key_type> keys, 
               thrust::device_ptr<mapped_type> results, 
               uint32_t numKeys, viewT view) {
  constexpr uint32_t BLOCK_SIZE = 128;
  constexpr uint32_t TILE_SIZE = BLOCK_SIZE * view.insertGran;
  uint32_t tid = TILE_SIZE * blockIdx.x + threadIdx.x;

  for(auto i = 0; i < TILE_SIZE; i += BLOCK_SIZE) {
    auto idx = tid + i;
    if(idx >= numKeys) {
      break;
    }
    auto key = keys[idx];
    auto found = view.find(key);
    //results[idx] = found->second;
  }
}



// cooperative group based search kernel
template<uint32_t tileSize, typename key_type, typename mapped_type, typename viewT>
__global__ void searchKeySetCG(thrust::device_ptr<key_type> keys, 
                               thrust::device_ptr<mapped_type> results, 
                               uint64_t numKeys, viewT view) {
  auto tile = cg::tiled_partition<tileSize>(cg::this_thread_block());
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid >= tileSize * numKeys) {
    return;
  }

  uint64_t keyIdx = tid / tileSize;
  auto found = view.findCG( tile, keys[keyIdx]);
  if(tile.thread_rank() == 0 && found != view.end()) {
    //results[keyIdx] = found->second;
  }
}