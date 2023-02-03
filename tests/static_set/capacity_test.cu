/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Static set capacity", "")
{
  constexpr std::size_t num_keys{400};
  using Key = int32_t;

  SECTION("Static extent must be evaluated at compile time.")
  {
    auto constexpr gold_capacity = 422;  // 211 x 2

    using extent_type = cuco::experimental::extent<std::size_t, num_keys>;
    cuco::experimental::static_set<Key, extent_type> set{extent_type{},
                                                         cuco::sentinel::empty_key<Key>{-1}};
    auto const capacity = set.capacity();
    STATIC_REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref();
    auto const ref_capacity = ref.capacity();
    STATIC_REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto constexpr gold_capacity = 422;  // 211 x 2

    cuco::experimental::static_set<Key> set{num_keys, cuco::sentinel::empty_key<Key>{-1}};
    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref();
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("Static extent must be evaluated at compile time.")
  {
    auto constexpr gold_capacity = 412;  // 103 x 2 x 2

    using extent_type = cuco::experimental::extent<std::size_t, num_keys>;
    using probe       = cuco::experimental::linear_probing<2, cuco::murmurhash3_32<Key>>;
    auto set          = cuco::experimental::
      static_set<Key, extent_type, cuda::thread_scope_device, thrust::equal_to<Key>, probe>{
        extent_type{},
        cuco::empty_key<Key>{-1},
        thrust::equal_to<Key>{},
        probe{cuco::murmurhash3_32<Key>{}}};

    REQUIRE(set.capacity() == gold_capacity);

    auto const capacity = set.capacity();
    STATIC_REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref();
    auto const ref_capacity = ref.capacity();
    STATIC_REQUIRE(ref_capacity == gold_capacity);
  }

  SECTION("Dynamic extent is evaluated at run time.")
  {
    auto constexpr gold_capacity = 412;  // 103 x 2 x 2

    using probe = cuco::experimental::linear_probing<2, cuco::murmurhash3_32<Key>>;
    auto set    = cuco::experimental::static_set<Key,
                                              cuco::experimental::extent<std::size_t>,
                                              cuda::thread_scope_device,
                                              thrust::equal_to<Key>,
                                              probe>{num_keys,
                                                     cuco::empty_key<Key>{-1},
                                                     thrust::equal_to<Key>{},
                                                     probe{cuco::murmurhash3_32<Key>{}}};

    auto const capacity = set.capacity();
    REQUIRE(capacity == gold_capacity);

    auto ref                = set.ref();
    auto const ref_capacity = ref.capacity();
    REQUIRE(ref_capacity == gold_capacity);
  }
}
