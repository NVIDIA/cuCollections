namespace cuco {



template<typename Key, typename Value, cuda::thread_scope Scope>
dynamic_map<Key, Value, Scope>::dynamic_map(std::size_t capacity, Key empty_key_sentinel, Value empty_value_sentinel) :
  empty_key_sentinel_(empty_key_sentinel),
  empty_value_sentinel_(empty_value_sentinel),
  num_submaps_(0),
  num_elements_(0),
  min_insert_size_(10'000),
  max_load_factor_(0.60) {
  for(auto i = 0; i < MAX_NUM_SUBMAPS_; ++i) {
    submap_caps_[i] = capacity;
  }

  // initialize and place first submap
  auto submap = new static_map<Key, Value, Scope>{submap_caps_[0], empty_key_sentinel, empty_value_sentinel};
  submaps_[0] = submap;
  num_submaps_++;
}



template<typename Key, typename Value, cuda::thread_scope Scope>
dynamic_map<Key, Value, Scope>::~dynamic_map() {
  for(auto i = 0; i < num_submaps_; ++i) {
    delete submaps_[i];
  }
}



template<typename Key, typename Value, cuda::thread_scope Scope>
void dynamic_map<Key, Value, Scope>::resize(std::size_t num_to_insert) {

  auto final_num_elements = num_elements_ + num_to_insert;
  auto num_elements_remaining = final_num_elements;
  auto submap_idx = 0;
  while(num_elements_remaining > 0) {
    auto submap_cap = submap_caps_[submap_idx];
    
    // if the submap does not exist yet, create it
    if(submap_idx >= num_submaps_) {
      auto submap =  
        new static_map<Key, Value, Scope>{submap_cap, empty_key_sentinel_, empty_value_sentinel_};
      submaps_[submap_idx] = submap;
      num_submaps_++;
    }

    num_elements_remaining -= max_load_factor_ * submap_cap - min_insert_size_;
    submap_idx++;
  }
}




template<typename Key, typename Value, cuda::thread_scope Scope>
template <typename InputIt, typename Hash, typename KeyEqual>
void dynamic_map<Key, Value, Scope>::insert(
  InputIt first, InputIt last, Hash hash, KeyEqual key_equal) {
  
  auto num_to_insert = std::distance(first, last);
  resize(num_to_insert);

  auto submap_idx = 0;
  while(num_to_insert > 0) {
    auto& submap = submaps_[submap_idx];
    auto capacity_remaining = max_load_factor_ * submap.get_capacity() - submap.get_size();
    if(capacity_remaining >= min_insert_size_) {
     // insert into submap submap_idx 
    }
    
    submap_idx++;
  }

}


}