#include "/home/nico/Documents/hashmap/SlabHash/src/gpu_hash_table.cuh"
#include <cuco/legacy_static_map.cuh>
#include <cuco/static_map.cuh>
#include <single_value_hash_table.cuh>
#include <cuco/dynamic_map.cuh>

#include <thrust/device_vector.h>
#include <benchmark/benchmark.h>
#include <synchronization.hpp>
#include <iostream>
#include <random>

enum class dist_type {
  UNIQUE,
  UNIQUE_NONE,
  UNIFORM,
  GAUSSIAN
};

template<dist_type Dist, typename Key, typename OutputIt>
static void generate_keys(OutputIt output_begin, OutputIt output_end) {
  auto num_keys = std::distance(output_begin, output_end);
  
  std::random_device rd;
  std::mt19937 gen{rd()};

  switch(Dist) {
    case dist_type::UNIQUE:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i + 1;
      }
      shuffle(output_begin, output_end, std::default_random_engine(15));
      break;
    case dist_type::UNIQUE_NONE:
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = i + 1 + num_keys;
      }
      shuffle(output_begin, output_end, std::default_random_engine(10));
      break;
    case dist_type::UNIFORM:
      // only works for Key = int32_t  
      for(auto i = 0; i < num_keys; ++i) {
        uint_fast32_t elem = gen();
        int32_t temp;
        std::memcpy(&temp, &elem, sizeof(int32_t)); // copy bits to int32_t
        temp = temp & 0x7FFFFFFF; // clear sign bit
        output_begin[i] = temp;
      }
      break;
    case dist_type::GAUSSIAN:
      std::normal_distribution<> dg{1e9, 1e7};
      for(auto i = 0; i < num_keys; ++i) {
        output_begin[i] = std::abs(static_cast<long>(dg(gen)));
      }
      break;
  }
}

template <typename Key, typename Value, dist_type Dist>
void slabhash_max_lf() {
    
 using map_type = gpu_hash_table<Key, Value, SlabHashTypeT::ConcurrentMap>;
  
  std::size_t num_keys = 1<<29;
  uint32_t base_size = 1.1 * (1<<30);
  std::size_t slab_size = 128;
  std::size_t num_buckets = base_size / slab_size;
  int64_t device_idx = 0;
  int64_t seed = 12;
  
  std::vector<Key> h_keys( num_keys );
  
  generate_keys<Dist, Key>(h_keys.begin(), h_keys.end());
  std::vector<Value> h_values (h_keys);

  ///*
  auto batch_size = 1e5;
  map_type map{num_keys, num_buckets, device_idx, seed, true, true, false};
  for(uint32_t i = 0; i < num_keys; i += batch_size) {
    float k = map.hash_build_with_unique_keys(h_keys.data() + i, 
                                                  h_values.data() + i, batch_size);
    std::cout << k << std::endl;
    std::cout << "lf " << static_cast<float>(i) / (1<<28)  << std::endl;
  }
  //*/

  //map.hash_search
  
}

int main() {
  slabhash_max_lf<int32_t, int32_t, dist_type::UNIQUE>();
  //slabhash_max_lf<int32_t, int32_t, dist_type::UNIFORM>();
}