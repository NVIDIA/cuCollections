/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuco/static_set.cuh>

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <thrust/device_vector.h>

#include <iostream>

/**
 * @file mapping_table_example.cu
 * @brief Demonstrates usage of the static_set "bulk" host APIs.
 *
 * The bulk APIs are only invocable from the host and are used for doing operations like `insert` or
 * `contains` on a set of keys.
 *
 */

template <typename Pointer>
struct my_equal {
  my_equal(Pointer data) : _data{data} {}
  __device__ constexpr bool operator()(int32_t lhs, int32_t rhs) const
  {
    return cuda::std::get<0>(*(_data + lhs)) == cuda::std::get<0>(*(_data + rhs)) and
           cuda::std::get<1>(*(_data + lhs)) == cuda::std::get<1>(*(_data + rhs)) and
           cuda::std::get<2>(*(_data + lhs)) == cuda::std::get<2>(*(_data + rhs)) and
           cuda::std::get<3>(*(_data + lhs)) == cuda::std::get<3>(*(_data + rhs));
  }
  Pointer _data;
};

template <typename Pointer>
struct my_hasher {
  my_hasher(Pointer data) : _data{data} {}
  __device__ auto operator()(int32_t index) const { return cuda::std::get<0>(_data[index]); }
  Pointer _data;
};

int main(void)
{
  using Key = cuda::std::tuple<uint32_t, char, bool, cuda::std::array<double, 4UL>>;
  auto const data =
    std::vector<Key>{cuda::std::tuple{11u, 'a', true, cuda::std::array{1., 2., 3., 4.}},
                     cuda::std::tuple{11u, 'a', true, cuda::std::array{1., 2., 3., 4.}},
                     cuda::std::tuple{22u, 'b', true, cuda::std::array{5., 6., 7., 8.}},
                     cuda::std::tuple{11u, 'a', true, cuda::std::array{5., 6., 7., 8.}},
                     cuda::std::tuple{11u, 'a', false, cuda::std::array{1., 2., 3., 4.}}};
  auto const size = data.size();
  thrust::device_vector<Key> d_data{data};

  using ActualKey = int32_t;

  ActualKey constexpr empty_key_sentinel = -1;

  auto const data_ptr = d_data.data().get();

  auto set = cuco::static_set{
    cuco::extent<std::size_t>{size * 2},
    cuco::empty_key{empty_key_sentinel},
    my_equal{data_ptr},
    cuco::linear_probing<1, my_hasher<Key const*>>{my_hasher<Key const*>{data_ptr}}};

  auto const actual_keys = thrust::device_vector<ActualKey>{0, 1, 2, 3, 4};
  set.insert(actual_keys.begin(), actual_keys.end());

  auto unique_keys           = thrust::device_vector<ActualKey>(size);
  auto const unique_keys_end = set.retrieve_all(unique_keys.begin());
  auto const num             = std::distance(unique_keys.begin(), unique_keys_end);

  std::cout << "There are " << num << " unique elements:\n";
  for (std::size_t i = 0; i < num; ++i) {
    auto const element = data[unique_keys[i]];
    std::cout << "[" << cuda::std::get<0>(element) << ", " << cuda::std::get<1>(element) << ", "
              << cuda::std::get<2>(element) << ", "
              << "[" << cuda::std::get<3>(element)[0] << ", " << cuda::std::get<3>(element)[1]
              << ", " << cuda::std::get<3>(element)[2] << ", " << cuda::std::get<3>(element)[3]
              << "]]\n";
  }

  return 0;
}
