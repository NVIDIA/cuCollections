// insert kernel
template<typename key_type, typename mapped_type, typename value_type, typename viewT>
__global__ void insertKeySet(thrust::device_ptr<key_type> keys, 
                             thrust::device_ptr<mapped_type> values,
                             uint32_t *totalNumSuccesses,
                             uint32_t numKeys, uint32_t submapIdx, viewT view) { 
    constexpr uint32_t BLOCK_SIZE = 128;
    constexpr uint32_t TILE_SIZE = BLOCK_SIZE * view.insertGran;
    uint32_t tid = TILE_SIZE * blockIdx.x + threadIdx.x;

    typedef cub::BlockReduce<uint32_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    
    uint32_t numSuccess = 0;
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
    uint32_t blockNumSuccesses = BlockReduce(tempStorage).Sum(numSuccess);
    if(threadIdx.x == 0) {
        atomicAdd(totalNumSuccesses, blockNumSuccesses);
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
