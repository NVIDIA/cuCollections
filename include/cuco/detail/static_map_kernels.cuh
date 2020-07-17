namespace cg = cooperative_groups;



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
template<uint32_t block_size,
         typename InputIt,
         typename atomicT,
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void insertKernel(InputIt first,
                             InputIt last,
                             atomicT* num_successes,
                             viewT view,
                             Hash hash,
                             KeyEqual key_equal) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;
  
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it = first + tid;
  
  while(it < last) {
    auto insert_pair = *it;
    auto res = view.insert(insert_pair, hash, key_equal);
    if(res.second) {
      thread_num_successes++;
    }
    it += gridDim.x * blockDim.x;
  }

  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
  std::size_t block_num_successes = BlockReduce(temp_storage).Sum(thread_num_successes);
  if(threadIdx.x == 0) {
    *num_successes += block_num_successes;
  }
}



// insert kernel
template<uint32_t block_size,
         uint32_t tile_size,
         typename InputIt,
         typename atomicT,
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void insertKernel(InputIt first,
                             InputIt last,
                             atomicT* num_successes,
                             viewT view,
                             Hash hash,
                             KeyEqual key_equal) {
  typedef cub::BlockReduce<std::size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t thread_num_successes = 0;

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto it = first + tid / tile_size;
  
  while(it < last) {
    auto insert_pair = *it;
    auto res = view.insert(tile, insert_pair, hash, key_equal);
    if(tile.thread_rank() == 0 && res.second) {
      thread_num_successes++;
    }
    it += (gridDim.x * blockDim.x) / tile_size;
  }
  
  // compute number of successfully inserted elements for each block
  // and atomically add to the grand total
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
                           viewT view,
                           Hash hash,
                           KeyEqual key_equal) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid;
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = view.find(key, hash, key_equal);
    *(output_begin + key_idx) = found->second.load(cuda::std::memory_order_relaxed);
    key_idx += gridDim.x * blockDim.x;
  }
}



// find kernel
template<uint32_t tile_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void findKernel(InputIt first,
                           InputIt last,
                           OutputIt output_begin,
                           viewT view,
                           Hash hash,
                           KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = view.find(tile, key, hash, key_equal);
    *(output_begin + key_idx) = found->second.load(cuda::std::memory_order_relaxed);
    key_idx += (gridDim.x * blockDim.x) / tile_size;
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
                               viewT view,
                               Hash hash,
                               KeyEqual key_equal) {
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid;
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = view.contains(key, hash, key_equal);
    *(output_begin + key_idx) = found;
    key_idx += gridDim.x * blockDim.x;
  }
}



// contains kernel
template<uint32_t tile_size,
         typename InputIt, typename OutputIt, 
         typename viewT,
         typename Hash, 
         typename KeyEqual>
__global__ void containsKernel(InputIt first,
                               InputIt last,
                               OutputIt output_begin,
                               viewT view,
                               Hash hash,
                               KeyEqual key_equal) {
  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto key_idx = tid / tile_size;
  
  while(first + key_idx < last) {
    auto key = *(first + key_idx);
    auto found = view.contains(tile, key, hash, key_equal);
    *(output_begin + key_idx) = found;
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }
}