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

#include <limits>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuco/static_multimap.cuh>

int main(void)
{
  int empty_key_sentinel   = -1;
  int empty_value_sentinel = -1;

  // Constructs a map with 100,000 slots using -1 and -1 as the empty key/value
  // sentinels. Note the capacity is chosen knowing we will insert 50,000 keys,
  // for an load factor of 50%.
  cuco::static_multimap<int, int> map{100'000, empty_key_sentinel, empty_value_sentinel};

  thrust::device_vector<thrust::pair<int, int>> pairs(50'000);

  // Create a sequence of pairs. Eeach key corresponds to two pairs.
  // E.g., {{0,0}, {1,1}, ... {0,25'000}, {1, 25'001}, ...}
  thrust::transform(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(pairs.size()),
                    pairs.begin(),
                    [] __device__(auto i) { return thrust::make_pair(i % (50'000 / 2), i); });

  // Inserts all pairs into the map
  map.insert(pairs.begin(), pairs.end());

  // Sequence of keys {0, 1, 2, ...}
  // thrust::device_vector<int> keys_to_find(50'000);
  // thrust::sequence(keys_to_find.begin(), keys_to_find.end(), 0);
  // thrust::device_vector<int> found_values(50'000);

  // Finds all keys {0, 1, 2, ...} and stores associated values into `found_values`
  // If a key `keys_to_find[i]` doesn't exist, `found_values[i] == empty_value_sentinel`
  // map.find(keys_to_find.begin(), keys_to_find.end(), found_values.begin());

  return 0;
}
