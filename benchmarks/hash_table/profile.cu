#include <benchmark/benchmark.h>

#include <cuco/static_map.cuh>
#include <cuco/dynamic_map.cuh>
#include "../nvtx3.hpp"

#include <thrust/for_each.h>
#include <iostream>
#include <fstream>


template <typename Key, typename Value>
static void cuco_search_all() {
  using map_type = cuco::static_map<Key, Value>;
  
  std::size_t num_keys = 100'000'000;
  float occupancy = 0.90;
  std::size_t size = num_keys / occupancy;

  map_type map{size, -1, -1};
  
  auto view = map.get_device_mutable_view();


  std::vector<int> h_keys( num_keys );
  std::vector<int> h_values( num_keys );
  std::vector<cuco::pair_type<int, int>> h_pairs ( num_keys );
  std::vector<int> h_results (num_keys);
  
  for(auto i = 0; i < num_keys; ++i) {
    h_keys[i] = i;
    h_pairs[i] = cuco::make_pair(i, i);
  }

  thrust::device_vector<int> d_keys( h_keys ); 
  thrust::device_vector<int> d_results( num_keys);
  thrust::device_vector<cuco::pair_type<int, int>> d_pairs( h_pairs );
  
  {
    nvtx3::thread_range r{"cuCo insert"};
    map.insert(d_pairs.begin(), d_pairs.end());
  }
  
  {
    nvtx3::thread_range r{"cuCo search"};
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
  }


}



template <typename Key, typename Value>
static void dynamic_search_all() {
  using map_type = cuco::dynamic_map<Key, Value>;
  
  std::size_t num_keys = 100'000'000;
  float occupancy = 0.50;
  std::size_t size = num_keys / occupancy;

  map_type map{size, -1, -1};
  
  std::vector<int> h_keys( num_keys );
  std::vector<int> h_values( num_keys );
  std::vector<cuco::pair_type<int, int>> h_pairs ( num_keys );
  std::vector<int> h_results (num_keys);
  
  for(auto i = 0; i < num_keys; ++i) {
    h_keys[i] = i;
    h_pairs[i] = cuco::make_pair(i, i);
  }

  thrust::device_vector<int> d_keys( h_keys ); 
  thrust::device_vector<int> d_results( num_keys);
  thrust::device_vector<cuco::pair_type<int, int>> d_pairs( h_pairs );
  
  {
    nvtx3::thread_range r{"cuCo insert"};
    map.insert(d_pairs.begin(), d_pairs.end());
  }
  
  {
    nvtx3::thread_range r{"cuCo search"};
    map.find(d_keys.begin(), d_keys.end(), d_results.begin());
  }

  thrust::copy(d_results.begin(), d_results.end(), h_results.begin());

  for(auto i = 0; i < num_keys; ++i) {
    if(h_results[i] != h_values[i]) {
      std::cout << "Key-value mismatch at index " << i << std::endl;
      std::cout << h_values[i] << std::endl;
      break;
    }
  }

}



int main() {
  for(auto i = 0; i < 1; ++i) {
    //cuco_search_all<int32_t, int32_t>();
    dynamic_search_all<int32_t, int32_t>();
  }
}