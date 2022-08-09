/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <utils.hpp>

#include <cuco/allocator.hpp>
#include <cuco/detail/storage.cuh>

#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE_SIG("Storage tests",
                       "",
                       ((typename Key, typename Value), Key, Value),
                       (int32_t, int32_t),
                       (int32_t, int64_t),
                       (int64_t, int64_t))
{
  constexpr std::size_t size{1'000};

  using allocator_type = cuco::cuda_allocator<char>;
  auto allocator       = allocator_type{};

  SECTION("Allocate array of pairs with AoS storage.")
  {
    auto s = cuco::detail::aos_storage<cuco::pair<Key, Value>, allocator_type>(size, allocator);
    auto const res_size = s.capacity();

    REQUIRE(res_size == size);
  }

  SECTION("Allocate array of keys with AoS storage.")
  {
    auto s              = cuco::detail::aos_storage<Key, allocator_type>(size, allocator);
    auto const res_size = s.capacity();

    REQUIRE(res_size == size);
  }
}
