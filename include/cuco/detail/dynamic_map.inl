namespace cuco {



template<typename Key, typename Value, cuda::thread_scope Scope>
dynamic_map<Key, Value, Scope>::dynamic_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel) :
  empty_key_sentinel_(empty_key_sentinel),
  empty_value_sentinel_(empty_value_sentinel),
  num_elements_(0),
  min_insert_size_(10'000),
  max_load_factor_(0.60) {
  for(auto i = 0; i < MAX_NUM_SUBMAPS_; ++i) {
    submap_caps_.push_back(capacity);
  }

  // initialize and place first submap
  auto submap = new static_map<Key, Value, Scope>{submap_caps_[0], empty_key_sentinel, empty_value_sentinel};
  submaps_.push_back(submap);

  // initialize device views
  submap_views_.push_back(submap->get_device_view());
  submap_mutable_views_.push_back(submap->get_device_mutable_view());
    
  CUCO_CUDA_TRY(cudaMallocManaged(&num_successes_, sizeof(atomic_ctr_type)));
}



template<typename Key, typename Value, cuda::thread_scope Scope>
dynamic_map<Key, Value, Scope>::~dynamic_map() {
  for(auto i = 0; i < submaps_.size(); ++i) {
    delete submaps_[i];
  }

  CUCO_CUDA_TRY(cudaFree(num_successes_));
}



template<typename Key, typename Value, cuda::thread_scope Scope>
void dynamic_map<Key, Value, Scope>::reserve(std::size_t n) {

  int64_t num_elements_remaining = n;
  auto submap_idx = 0;
  while(num_elements_remaining > 0) {
    auto submap_cap = submap_caps_[submap_idx];
    
    // if the submap does not exist yet, create it
    if(submap_idx >= submaps_.size()) {
      auto submap =  
        new static_map<Key, Value, Scope>{submap_cap, empty_key_sentinel_, empty_value_sentinel_};
      submaps_.push_back(submap);
      
      submap_views_.push_back(submap->get_device_view());
      submap_mutable_views_.push_back(submap->get_device_mutable_view());
    }

    num_elements_remaining -= max_load_factor_ * submap_cap - min_insert_size_;
    submap_idx++;
  }
}



template<typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope>::insert(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal) {
  
  std::size_t num_to_insert = std::distance(first, last);
  reserve(num_elements_ + num_to_insert);

  thrust::device_vector<view_type> d_submap_views( submap_views_ );
  thrust::device_vector<mutable_view_type> d_submap_mutable_views( submap_mutable_views_ );

  uint32_t submap_idx = 0;
  while(num_to_insert > 0) {
    auto submap = submaps_[submap_idx];
    std::size_t capacity_remaining = max_load_factor_ * submap->get_capacity() - submap->get_size();
    // If we are tying to insert some of the remaining keys into this submap, we can insert 
    // only if we meet the minimum insert size.
    if(capacity_remaining >= min_insert_size_) {
      *num_successes_ = 0;
      CUCO_CUDA_TRY(cudaMemPrefetchAsync(num_successes_, sizeof(atomic_ctr_type), 0));
      
      auto n = std::min(capacity_remaining, num_to_insert);
      auto const block_size = 128;
      auto const stride = 1;
      auto const tile_size = 1;
      auto const grid_size = (tile_size * n + stride * block_size - 1) /
                             (stride * block_size);

      insertKernel<block_size, cuco::pair_type<key_type, mapped_type>>
      <<<grid_size, block_size>>>
      (first, first + n,
       d_submap_views.data().get(),
       d_submap_mutable_views.data().get(),
       num_successes_, submap_idx, hash, key_equal);
      CUCO_CUDA_TRY(cudaDeviceSynchronize());

      std::size_t h_num_successes = num_successes_->load(cuda::std::memory_order_relaxed);
      submap->incr_size(h_num_successes);
      num_elements_ += h_num_successes;
      first += n;
      num_to_insert -= n;
    }
    
    submap_idx++;
  }
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename OutputIt, 
          typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope>::find(
  InputIt first, InputIt last, OutputIt output_begin,
  Hash hash, KeyEqual key_equal) noexcept {
  
  thrust::device_vector<view_type> d_submap_views( submap_views_ );
  
  auto num_keys = std::distance(first, last);
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 1;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                          (stride * block_size);

  findKernel<>
  <<<grid_size, block_size>>>
  (first, last, output_begin,
   d_submap_views.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());    
}



template <typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename OutputIt, 
          typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope>::contains(
  InputIt first, InputIt last, OutputIt output_begin,
  Hash hash, KeyEqual key_equal) noexcept {
  
  thrust::device_vector<view_type> d_submap_views( submap_views_ );
  
  auto num_keys = std::distance(first, last);
  auto const block_size = 128;
  auto const stride = 1;
  auto const tile_size = 1;
  auto const grid_size = (tile_size * num_keys + stride * block_size - 1) /
                          (stride * block_size);
  
  containsKernel<>
  <<<grid_size, block_size>>>
  (first, last, output_begin,
   d_submap_views.data().get(), submaps_.size(), hash, key_equal);
  CUCO_CUDA_TRY(cudaDeviceSynchronize());    
}



}