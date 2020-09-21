/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <limits>

#include <cuco/static_map.cuh>

int main(void)
{
  int empty_key_sentinel   = std::numeric_limits<int>::max();
  int empty_value_sentinel = std::numeric_limits<int>::max();
  cuco::static_map<int, int> my_map{100'000, empty_key_sentinel, empty_value_sentinel};
  thrust::device_vector<thrust::pair<int, int>> pairs(50'000);
  my_map.get_device_view();
  my_map.insert(pairs.begin(), pairs.end());
  return 0;
}
