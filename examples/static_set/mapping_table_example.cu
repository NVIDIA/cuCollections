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

int main(void)
{
  auto const data = thrust::device_vector{
    cuda::std::tuple{11, "string", true, cuda::std::array{1., 2., 3., 4.}},
    cuda::std::tuple{11, "string", true, cuda::std::array{1., 2., 3., 4.}},
    cuda::std::tuple{22, "I'm a looooooooooooooong string", true, cuda::std::array{5., 6., 7., 8.}},
    cuda::std::tuple{11, "string", true, cuda::std::array{1., 2., 3., 4.}},
    cuda::std::tuple{11, "string", false, cuda::std::array{1., 2., 3., 4.}}};

  using Key = int32_t;

  // Empty slots are represented by reserved "sentinel" values. These values should be selected such
  // that they never occur in your input data.
  Key constexpr empty_key_sentinel = -1;

  std::cout << data.size() << std::endl;

  return 0;
}
