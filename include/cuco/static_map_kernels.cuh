template<typename atomic_key_type, typename atomic_mapped_type, typename Key, typename Value, typename pair_atomic_type>
__global__ void initializeKernel(
    pair_atomic_type* const __restrict__ slots, Key k,
    Value v, std::size_t size) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  while (tid < size) {
    new (&slots[tid].first) atomic_key_type{k};
    new (&slots[tid].second) atomic_mapped_type{v};
    tid += gridDim.x * blockDim.x;
  }
}

// insert kernel
template<typename InputIt,
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void insertKernel(InputIt first,
                             InputIt last,
                             viewT view,
                             Hash hash,
                             KeyEqual key_equal) {
  
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it = first + tid;
  
  while(it < last) {
    auto insert_pair = *it;
    view.insert(insert_pair);
    it += gridDim.x * blockDim.x;
  }
}

// search kernel
template<typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void findKernel(InputIt first,
                           InputIt last,
                           OutputIt output_begin,
                           viewT view,
                           Hash hash,
                           KeyEqual key_equal) {
  
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it = first + tid;
  
  while(it < last) {
    auto key = *it;
    auto found = view.find(key, hash, key_equal);
    *(output_begin + tid) = found->second;
    it += gridDim.x * blockDim.x;
  }
}