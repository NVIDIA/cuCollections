/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

#include <cuco/static_multimap.cuh>

#include <cuda/iterator>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <iostream>

int main(void)
{
  using key_type   = int;
  using value_type = int;

  key_type empty_key_sentinel     = -1;
  value_type empty_value_sentinel = -1;

  constexpr std::size_t N = 50'000;

  // Constructs a multimap with 100,000 slots using -1 and -1 as the empty key/value
  // sentinels. Note the capacity is chosen knowing we will insert 50,000 keys,
  // for an load factor of 50%.
  cuco::static_multimap<key_type, value_type> map{
    N * 2, cuco::empty_key{empty_key_sentinel}, cuco::empty_value{empty_value_sentinel}};

  thrust::device_vector<cuco::pair<key_type, value_type>> pairs(N);

  // Create a sequence of pairs. Eeach key has two matches.
  // E.g., {{0,0}, {1,1}, ... {0,25'000}, {1, 25'001}, ...}
  thrust::transform(
    cuda::make_counting_iterator<int>(0),
    cuda::make_counting_iterator<int>(pairs.size()),
    pairs.begin(),
    [] __device__(auto i) { return cuco::pair<key_type, value_type>{i % (N / 2), i}; });

  // Inserts all pairs into the map
  map.insert(pairs.begin(), pairs.end());

  // Sequence of probe keys {0, 1, 2, ... 24'999}
  // Each key should have 2 matches in the map
  thrust::device_vector<key_type> keys_to_find(N / 2);
  thrust::sequence(keys_to_find.begin(), keys_to_find.end(), 0);

  // Check that keys are contained in the map
  thrust::device_vector<bool> contained(N / 2);
  map.contains(keys_to_find.begin(), keys_to_find.end(), contained.begin());

  // Verify all keys are found
  auto const num_found = thrust::count(contained.begin(), contained.end(), true);

  if (num_found == N / 2) {
    std::cout << "Success! All " << N / 2 << " unique keys found in the multimap." << std::endl;
    std::cout << "Each key has 2 duplicate values, for a total of " << N << " pairs." << std::endl;
  }

  return 0;
}
