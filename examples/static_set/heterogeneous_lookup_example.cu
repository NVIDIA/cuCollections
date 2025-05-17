/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuda/std/string_view>
#include <thrust/device_vector.h>

#include <iostream>

/**
 * @file heterogeneous_lookup_example.cu
 *
 * @brief Demonstrates how to use hash set as a lookup table of the original data with string keys
 *
 * @note This example is for demonstration purposes only. It is not intended to show the most
 * performant way to do the example algorithm.
 */

/**
 * @brief User-defined key equal to compare two keys
 */
struct my_equal {
  my_equal(cuda::std::string_view* data) : _data{data} {}
  /**
   * @brief Checks if two keys are identical based on their indices in the
   * original data array
   */
  __device__ constexpr bool operator()(int32_t lhs, int32_t rhs) const
  {
    return this->operator()(_data[lhs], rhs);
  }

  __device__ constexpr bool operator()(cuda::std::string_view lhs_str, int32_t rhs) const
  {
    auto rhs_str = _data[rhs];

    // First check if lengths are the same
    if (lhs_str.size() != rhs_str.size()) return false;

    // Then compare each character
    for (size_t i = 0; i < lhs_str.size(); ++i) {
      if (lhs_str[i] != rhs_str[i]) return false;
    }

    return true;
  }
  cuda::std::string_view* _data;
};

/**
 * @brief User-defined hash function to hash the original data based on its index
 *
 * @tparam T Original key type
 */
struct my_hasher {
  my_hasher(cuda::std::string_view* data) : _data{data} {}
  __device__ uint32_t operator()(int32_t index) const
  {
    auto str = _data[index];
    return this->operator()(str);
  }

  __device__ uint32_t operator()(cuda::std::string_view str) const
  {
    uint32_t hash = 0;
    // Naive string hash function
    hash = str[0] + str.size();

    return hash;
  }
  cuda::std::string_view* _data;
};

int main(void)
{
  // The original key type is a string, which is variable length and larger than 8 bytes
  using Key = std::string;

  auto input =
    thrust::device_vector<cuda::std::string_view>{"apple", "apple", "banana", "cherry", "apple"};

  auto const size = input.size();

  // The actual key type is an index type, `int32_t` is large enough to cover the whole input range
  // and 4-byte atomic CAS is more efficient than the 8-byte one.
  using ActualKey = int32_t;
  // `-1` is a valid sentinel value since one will never access `data[-1]`
  ActualKey constexpr empty_key_sentinel = -1;

  auto const data_ptr = input.data().get();
  auto set = cuco::static_set{cuco::extent<std::size_t>{size * 2},  // about 50% load factor
                              cuco::empty_key{empty_key_sentinel},
                              my_equal{data_ptr},
                              cuco::linear_probing<1, my_hasher>{my_hasher{data_ptr}}};

  // The actual keys are indices of elements
  auto const actual_keys = thrust::device_vector<ActualKey>{0, 1, 2, 3, 4};
  set.insert(actual_keys.begin(), actual_keys.end());

  auto query = thrust::device_vector<cuda::std::string_view>{"lychee", "cherry", "apple"};
  auto const query_size = query.size();

  auto contained = thrust::device_vector<bool>(query_size);
  set.contains(query.begin(), query.end(), contained.begin());

  for (auto const& it : contained) {
    std::cout << it << "\n";
  }

  return 0;
}