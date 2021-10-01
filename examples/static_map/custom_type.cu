/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

#include <cuco/static_map.cuh>

// User-defined key type
// Manual alignment required due to WAR libcu++ bug where cuda::atomic fails for underaligned types
struct alignas(8) key_pair_type {
  int32_t a;
  int32_t b;

  __host__ __device__ key_pair_type() {}
  __host__ __device__ key_pair_type(int32_t x) : a{x}, b{x} {}

  // Device equality operator is mandatory
  __device__ bool operator==(key_pair_type const& other) const
  {
    return a == other.a and b == other.b;
  }
};

// Key type larger than 8B only supported for sm_70 and up
struct key_triplet_type {
  int32_t a;
  int32_t b;
  int32_t c;

  __host__ __device__ key_triplet_type() {}
  __host__ __device__ key_triplet_type(int32_t x) : a{x}, b{x}, c{x} {}

  // Device equality operator is mandatory
  __device__ bool operator==(key_triplet_type const& other) const
  {
    return a == other.a and b == other.b and c == other.c;
  }
};

// User-defined value type
// Manual alignment required due to WAR libcu++ bug where cuda::atomic fails for underaligned types
struct alignas(8) custom_value_type {
  int32_t f;
  int32_t s;

  __host__ __device__ custom_value_type() {}
  __host__ __device__ custom_value_type(int32_t x) : f{x}, s{x} {}
};

// User-defined device hash callable
struct custom_hash {
  template <typename key_type>
  __device__ uint32_t operator()(key_type k)
  {
    return k.a;
  };
};

// User-defined device key equal callable
struct custom_key_equals {
  template <typename key_type>
  __device__ bool operator()(key_type const& lhs, key_type const& rhs)
  {
    return lhs.a == rhs.a;
  }
};

template <typename custom_key_type>
void run_example()
{
  constexpr std::size_t num_pairs = 80'000;

  // Set emtpy sentinels
  auto const empty_key_sentinel   = custom_key_type{-1};
  auto const empty_value_sentinel = custom_value_type{-1};

  thrust::device_vector<thrust::pair<custom_key_type, custom_value_type>> pairs(num_pairs);
  // Create a sequence of 80'000 pairs
  thrust::transform(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(num_pairs),
    pairs.begin(),
    [] __device__(auto i) { return thrust::make_pair(custom_key_type{i}, custom_value_type{i}); });

  // Construct a map with 100,000 slots using the given empty key/value sentinels. Note the
  // capacity is chosen knowing we will insert 80,000 keys, for an load factor of 80%.
  cuco::static_map<custom_key_type, custom_value_type> map{
    100'000, empty_key_sentinel, empty_value_sentinel};

  // Inserts all pairs into the map by using the custom hasher and custom equality callable
  map.insert(pairs.begin(), pairs.end(), custom_hash{}, custom_key_equals{});

  // Reproduce inserted keys
  auto insert_keys =
    thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0),
                                    [] __device__(auto i) { return custom_key_type{i}; });

  thrust::device_vector<bool> contained(num_pairs);

  // Determine if all the inserted keys can be found by using the same hasher and equality
  // function as `insert`. If a key `insert_keys[i]` doesn't exist, `contained[i] == false`.
  map.contains(
    insert_keys, insert_keys + num_pairs, contained.begin(), custom_hash{}, custom_key_equals{});
  // This will fail due to inconsistent hash and key equal.
  // map.contains(insert_keys, insert_keys + num_pairs, contained.begin());

  // All inserted keys are contained
  assert(
    thrust::all_of(contained.begin(), contained.end(), [] __device__(auto const& b) { return b; }));
}

int main(void)
{
  constexpr int volta_major_number = 7;

  // Retrieve major compute capability version number
  int dev_id, cap_major;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&cap_major, cudaDevAttrComputeCapabilityMajor, dev_id);

  // Run 8B-key example on Pascal
  if (cap_major < volta_major_number) {
    run_example<key_pair_type>();
  }
  // 12B-key example on sm_70 and up
  else {
    run_example<key_triplet_type>();
  }

  return 0;
}
